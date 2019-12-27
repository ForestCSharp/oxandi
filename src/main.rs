#![cfg_attr(
    not(any(feature = "dx12", feature = "metal", feature = "vulkan")),
    allow(unused)
)]

use std::{
    sync::{Arc, RwLock},
    path::PathBuf, 
    time::Instant,
    collections::HashMap,
};

#[macro_use]
extern crate derive_deref;

use rendy::{
    wsi::Surface,
    command::{QueueId, RenderPassEncoder, Families},
    factory::{Config, Factory},
    graph::{render::*, GraphBuilder, GraphContext, NodeBuffer, NodeImage, DescBuilder},
    hal::{self, device::Device as _, window::PresentMode},
    memory::Dynamic,
    resource::{Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle},
    init::winit::{
            event::{Event, WindowEvent, DeviceEvent, VirtualKeyCode},
            event_loop::{ControlFlow, EventLoop},
            window::WindowBuilder,
            dpi::PhysicalSize,
    },
    init::winit,
    init::AnyWindowedRendy,
};

extern crate nalgebra_glm as glm;

extern crate specs;
use specs::prelude::*;
#[macro_use]
extern crate specs_derive;

extern crate rhai;
use rhai::{RegisterFn, Scope};

#[cfg(feature = "dx12")]
pub type Backend = rendy::dx12::Backend;

#[cfg(feature = "metal")]
pub type Backend = rendy::metal::Backend;

#[cfg(feature = "vulkan")]
pub type Backend = rendy::vulkan::Backend;

//TODO: REMOVE and just extract directly from component data
pub struct CameraRenderData {
    pub position : glm::Vec3,
    pub target   : glm::Vec3,
    pub up       : glm::Vec3,
}

impl CameraRenderData {
    fn new() -> CameraRenderData {
        CameraRenderData {
            position : glm::vec3(0., 0., 1.),
            target   : glm::vec3(0., 0., 0.),
            up       : glm::vec3(0., 1., 0.),
        }
    }
}

pub struct Scene<B: hal::Backend> {
    pub specs_world : Arc<RwLock<World>>,
    pub camera_data : CameraRenderData,
    pub frames_in_flight : u32,
    pub phantom: std::marker::PhantomData<B>,
}

#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
struct UniformData {
    model_matrix : glm::Mat4,
    view_matrix  : glm::Mat4,
    proj_matrix  : glm::Mat4,
}

#[derive(Debug, Default)]
struct BasicRenderGroupDesc;

impl<B> RenderGroupDesc<B,Scene<B>> for BasicRenderGroupDesc
    where
        B: hal::Backend,
{
     fn build<'a>(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        queue: QueueId,
        aux: &Scene<B>,
        framebuffer_width: u32,
        framebuffer_height: u32,
        subpass: hal::pass::Subpass<'_, B>,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
    ) -> Result<Box<dyn RenderGroup<B, Scene<B>>>, hal::pso::CreationError> {
        return Ok(Box::new(BasicRenderGroup {
            framebuffer_size : (framebuffer_width, framebuffer_height),
            mesh_data : HashMap::new(),
        }));
    }
}

#[derive(Debug, Default)]
struct BasicRenderGroupMeshData<B : hal::Backend> {
    pub pipeline : B::GraphicsPipeline,
    pub uniform_buffers : Vec<Escape<Buffer<B>>>, // 1 per frame-in-flight
    pub skeleton_buffers : Vec<Escape<Buffer<B>>>,
    pub descriptor_sets : Vec<Escape<DescriptorSet<B>>>, // 1 per frame-in-flight
}

#[derive(Debug, Default)]
struct BasicRenderGroup<B : hal::Backend> {
    framebuffer_size : (u32, u32),
    mesh_data : HashMap<u32, BasicRenderGroupMeshData<B>>,
}

impl<B> BasicRenderGroup<B> 
    where B : hal::Backend,
{
    fn builder() -> DescBuilder<B, Scene<B>, BasicRenderGroupDesc>
    {
        BasicRenderGroupDesc::default().builder()
    }
}

impl<B> RenderGroup<B, Scene<B>> for BasicRenderGroup<B>
    where B: hal::Backend,
{   
    fn prepare(
        &mut self,
        factory: &Factory<B>,
        queue: QueueId,
        index: usize,
        subpass: hal::pass::Subpass<'_, B>,
        scene: &Scene<B>,
    ) -> PrepareResult {
        let specs_world = scene.specs_world.read().expect("Failed to read from scene.specs_world RwLock");
        let specs_entities = specs_world.entities();
        let specs_meshes = specs_world.read_storage::<MeshComponent<B>>();
        let specs_materials = specs_world.read_storage::<Material<B>>();
        let specs_transforms = specs_world.read_storage::<Transform>();

        for (entity, mesh, material, transform) in (&specs_entities, &specs_meshes, &specs_materials, &specs_transforms).join() {
            let id = entity.id();

            let uniform_size = std::mem::size_of::<UniformData>() as u64;
            let mut uniform_buffers = Vec::with_capacity(scene.frames_in_flight as usize);

            let skeleton_size = match mesh.gltf_model.skeletons.get(0) {
                    Some(skeleton) => Some(skeleton.bones.len() as u64 * std::mem::size_of::<GpuBone>() as u64),
                    None => None,
            };

            let mut skeleton_buffers = Vec::new();

            let mut descriptor_sets = Vec::with_capacity(scene.frames_in_flight as usize);

            for index in 0..scene.frames_in_flight as usize {
                uniform_buffers.push(factory
                    .create_buffer(
                        BufferInfo {
                            size: uniform_size,
                            usage: hal::buffer::Usage::UNIFORM,
                        },
                        Dynamic,
                    )
                    .unwrap()
                );

                if let Some(skeleton_size) = skeleton_size {
                    skeleton_buffers.push(factory
                        .create_buffer(
                            BufferInfo {
                                size: skeleton_size,
                                usage: hal::buffer::Usage::UNIFORM,
                            },
                            Dynamic,
                        )
                        .unwrap()
                    );
                }

                descriptor_sets.push( unsafe {
                    let set = factory
                        .create_descriptor_set(material.set_layouts[0].clone())
                        .unwrap();
                    factory.write_descriptor_sets(Some(hal::pso::DescriptorSetWrite {
                        set: set.raw(),
                        binding: 0,
                        array_offset: 0,
                        descriptors: Some(hal::pso::Descriptor::Buffer(
                            uniform_buffers[index].raw(),
                            Some(0) .. Some(uniform_size)
                        )),
                    }));
                    if let Some(skeleton_size) = skeleton_size {
                        factory.write_descriptor_sets(Some(hal::pso::DescriptorSetWrite {
                            set: set.raw(),
                            binding: 1,
                            array_offset: 0,
                            descriptors: Some(hal::pso::Descriptor::Buffer(
                                skeleton_buffers[index].raw(),
                                Some(0) .. Some(skeleton_size)
                            )),
                        }));
                    }
                    set
                });     
            }

            if !self.mesh_data.contains_key(&id) {
                self.mesh_data.insert(id, BasicRenderGroupMeshData {
                    pipeline : material.create_pipeline(subpass, factory),
                    uniform_buffers,
                    skeleton_buffers,
                    descriptor_sets,
                });
            }

            if let Some(mut mesh_data) = self.mesh_data.get_mut(&entity.id()) {
                unsafe {
                    let model_matrix = {
                        let mut model_matrix = glm::identity();
                        model_matrix = glm::translate(&model_matrix, &transform.position);
                        model_matrix *= glm::quat_to_mat4(&transform.rotation);
                        model_matrix *= glm::scale(&glm::identity(), &transform.scale);
                        model_matrix
                    };

                    let mut perspective_matrix = glm::perspective_zo(
                        self.framebuffer_size.0 as f32 / self.framebuffer_size.1 as f32,
                        degrees_to_radians(60.0f32),
                        100000.0,
                        0.01
                    );
                    perspective_matrix[(1,1)] *= -1.0;

                    factory.upload_visible_buffer(
                        &mut mesh_data.uniform_buffers[index], //access correct uni buffer
                        0, 
                        &[UniformData {
                            model_matrix :  model_matrix,
                            view_matrix  :  glm::look_at(
                                                &scene.camera_data.position,
                                                &scene.camera_data.target,
                                                &scene.camera_data.up,
                                            ),
                            proj_matrix  :  perspective_matrix.into(),
                        }]
                    ).unwrap();

                    if let Some(skeleton_buffer) = mesh_data.skeleton_buffers.get(index) {
                        factory.upload_visible_buffer(
                            &mut mesh_data.skeleton_buffers[index], //access correct uni buffer
                            0,
                            &mesh.gltf_model.skeletons[0].bones,
                        ).unwrap();
                    }
                }
            }
        }

        PrepareResult::DrawRecord
    }

    fn draw_inline(
        &mut self,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        _subpass: hal::pass::Subpass<'_, B>,
        scene: &Scene<B>,
    ) {
        let specs_world = scene.specs_world.read().expect("Failed to read from scene.specs_world RwLock");
        let specs_entities = specs_world.entities();
        let specs_meshes = specs_world.read_storage::<MeshComponent<B>>();
        let specs_materials = specs_world.read_storage::<Material<B>>();

        let viewport = hal::pso::Viewport {
			rect: hal::pso::Rect {
				x: 0,
				y: 0,
				w: self.framebuffer_size.0 as i16,
				h: self.framebuffer_size.1 as i16,
			},
			depth: 0.0..1.0,
		};
        
        unsafe {
            encoder.set_viewports(0, &[viewport.clone()]);
            encoder.set_scissors(0, &[viewport.rect]);
        }

        for (entity, mesh, material) in (&specs_entities, &specs_meshes, &specs_materials).join() {
            if let Some(mesh_data) = self.mesh_data.get(&entity.id()) {
                encoder.bind_graphics_pipeline(&mesh_data.pipeline);
                unsafe {
                    encoder.bind_graphics_descriptor_sets(
                        &material.pipeline_layout,
                        0,
                        Some(mesh_data.descriptor_sets[index].raw()),
                        std::iter::empty(),
                    );
                    encoder.bind_vertex_buffers(0, Some((mesh.vertex_buffer.raw(), 0)));
                    encoder.bind_index_buffer(mesh.index_buffer.raw(), 0, hal::IndexType::U32);
                    encoder.draw_indexed(0..mesh.num_indices, 0, 0..1);
                }
            }
        }
    }

    fn dispose(self: Box<Self>, factory: &mut Factory<B>, scene: &Scene<B>) {
        //TODO:
    }
}

mod specs_systems;
use specs_systems::{common::*, spatial::*, mesh::*};

mod gltf_loader;
use gltf_loader::*;

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn main() {

    env_logger::Builder::from_default_env()
    .init();

    let gltf_model = GltfModel::new("data/models/Running.glb");

    //Specs setup
    let specs_world = Arc::new(RwLock::new(World::new()));

    let anim_system = AnimationSystem::<Backend> {
        phantom : std::marker::PhantomData::<Backend>,
    };

    let specs_dispatcher = {
        let mut dispatcher = DispatcherBuilder::new()
            .with(InputMovementSystem, "input_movement_system", &[])
            .with(UpdateVelocitySystem, "acceleration_system", &[])
            .with(UpdatePositionSystem, "update_position_system", &["input_movement_system", "acceleration_system"])
            .with(UpdateRotationSystem, "update_rotation_system", &["input_movement_system"])
            .with(UpdateCameraSystem, "update_camera_system", &["input_movement_system"])
            .with(anim_system, "animation_system", &["input_movement_system"])
        .build();
        //NOTE: has to be done before creating entities in world (seems problematic)
        dispatcher.setup(&mut specs_world.write().unwrap());
        Arc::new(RwLock::new(dispatcher))
    };

    let config: Config = Default::default();

    let mut event_loop = EventLoop::new();

    let window_builder = WindowBuilder::new()
        .with_title("Oxandi")
        .with_inner_size((960, 640).into())
        .with_title("Oxandi");

    let rendy = AnyWindowedRendy::init_auto(&config, window_builder, &event_loop).unwrap();
    rendy::with_any_windowed_rendy!((rendy)
        (mut factory, mut families, surface, window) => {

            let mut build_graph = |size : PhysicalSize, surface : Surface<Backend>, scene : &Scene<Backend>, factory : &mut Factory<Backend>, families : &mut Families<Backend>| {
                
                //TODO: get frames in flight from surface?
                let mut graph_builder = GraphBuilder::<_, Scene<Backend>>::new().with_frames_in_flight(scene.frames_in_flight);
                let window_kind = hal::image::Kind::D2(size.width as u32, size.height as u32, 1, 1);

                let depth = graph_builder.create_image(
                    window_kind,
                    1,
                    hal::format::Format::D32Sfloat,
                    Some(hal::command::ClearValue{
                        depth_stencil : hal::command::ClearDepthStencil{depth: 0.0, stencil: 0},
                    }),
                );
    
                graph_builder.add_node(
                    BasicRenderGroup::builder()
                        .into_subpass()
                        .with_color_surface()
                        .with_depth_stencil(depth)
                        .into_pass()
                        .with_surface(
                            surface,
                            hal::window::Extent2D {
                                width: size.width as _,
                                height: size.height as _,
                            },
                            Some(hal::command::ClearValue {
                                color: hal::command::ClearColor {
                                    float32: [1.0, 1.0, 1.0, 1.0],
                                },
                            }
                        ),
                    ),
                );

                Some(graph_builder
                    .build(factory, families, scene)
                    .unwrap())
            };

            let mut scene = Scene {
                specs_world : specs_world.clone(),
                camera_data : CameraRenderData::new(),
                frames_in_flight : 3, //FIXME: (get this from somewhere else?)
                phantom : std::marker::PhantomData,
            };

            let mut size = window.inner_size().to_physical(window.hidpi_factor());
            let mut graph = build_graph(size, surface, &scene, &mut factory, &mut families);

            {
                let mut specs_world = specs_world.write().unwrap();
                specs_world.insert(DeltaTime(0.05)); //add a resource
                specs_world.insert(InputState::new());
                specs_world.insert(scene);
                specs_world.create_entity().with(Velocity(glm::vec3(2.0, 1.0, 3.0))).with(Transform::new()).build();
                specs_world.create_entity().with(Velocity(glm::vec3(200.0, 100.0, 300.0))).with(Transform::new()).build();

                specs_world.register::<MeshComponent<Backend>>(); //FIXME: can remove this once a system using MeshComponent is hooked up to world
                specs_world.register::<Material<Backend>>(); //FIXME: can remove this once a system using Material is hooked up to world
                
                specs_world.create_entity()
                    .with(Transform::new())
                    .with(MeshComponent::new("data/models/Cube.glb", &mut factory))
                    .with(Material::new("/data/shaders/basic.vert", "/data/shaders/basic.frag", &mut factory))
                    .build();
                
                for i in 0..1 {
                    let mut transform = Transform::new();
                    transform.position.x = 2.0 * i as f32;
                    transform.rotation = glm::quat(1.0, 0.0, 0.0, degrees_to_radians(-90.0));

                    specs_world.create_entity()
                        .with(MeshComponent::new("data/models/Running.glb", &mut factory))
                        .with(AnimComponent::new(0.5))
                        .with(Material::new("/data/shaders/skinned.vert", "/data/shaders/basic_2.frag", &mut factory))
                        .with(transform)
                        .with(Velocity(glm::vec3(0.05, 0.0, 0.0)))
                        //.with(AngularVelocity(glm::quat(0.0, 1.0, 1.0, 0.03))) //FIXME: lower quat values not slowing down rotation, why?
                        .build();
                }
                
                //Camera
                specs_world.create_entity()
                    .with(Transform::new())
                    .with(Velocity(glm::vec3(0.0, 0.0, 0.0)))
                    .with(Acceleration(glm::vec3(0.0, 0.0, 0.0)))
                    .with(Drag(10.0))
                    .with(TerminalVelocity(100.0))
                    .with(InputComponent { move_speed : 20.0 })
                    .with(ActiveCamera)
                    .build();
                
                specs_world.create_entity()
                    .with(Velocity(glm::vec3(2.0, 1.0, 3.0)))
                    .with(Transform::new())
                    .with(Acceleration(glm::vec3(2.0, 1.0, 2.0)))
                    .build();

                specs_world.create_entity().with(Transform::new()).build();
            }

            fn dispatch_world(dispatcher: Arc<RwLock<specs::Dispatcher>>, world : Arc<RwLock<specs::World>>) {
                dispatcher.write().unwrap().dispatch_par(&world.read().unwrap());
            }

            let mut rhai_engine = rhai::Engine::new();
            rhai_engine.register_fn("dispatch_world", dispatch_world);
            rhai_engine.register_type::<Arc<RwLock<specs::Dispatcher>>>();
            rhai_engine.register_type::<Arc<RwLock<specs::World>>>();
        
            let mut scope = Scope::new();
            scope.push(("dispatcher".to_string(), Box::new(specs_dispatcher.clone())));
            scope.push(("world". to_string(), Box::new(specs_world.clone())));

            let mut delta_time = Instant::now();
            let mut new_delta = (0., 0.);
            let mut needs_resize = false;

            event_loop.run(move |event, _, control_flow| {

                match event {
                    Event::WindowEvent { event, .. } => {
                        match event { 
                            WindowEvent::CloseRequested => { 
                                *control_flow = ControlFlow::Exit;
                            },
                            WindowEvent::KeyboardInput { device_id : _, input } => {
                                match input.virtual_keycode {
                                    Some(keycode) => {
                                        let specs_world = specs_world.read().unwrap();
                                        specs_world.write_resource::<InputState>().key_states.insert(keycode, input.state == winit::event::ElementState::Pressed);

                                        match keycode {
                                            VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                                            _ => {},
                                        }
                                    },
                                    _ => {},
                                }
                            },
                            WindowEvent::MouseInput { state, button, ..} => {
                                let pressed = state == winit::event::ElementState::Pressed;
                                let specs_world = specs_world.read().unwrap();
                                let mouse_button_states = &mut specs_world.write_resource::<InputState>().mouse_button_states;
                                match button {
                                    winit::event::MouseButton::Left   => mouse_button_states[0] = pressed,
                                    winit::event::MouseButton::Right  => mouse_button_states[1] = pressed,
                                    winit::event::MouseButton::Middle => mouse_button_states[2] = pressed,
                                    winit::event::MouseButton::Other(idx) => {
                                        if (idx as usize) < mouse_button_states.len() {
                                            mouse_button_states[idx as usize] = pressed;
                                        }
                                    }
                                }
                            },
                            WindowEvent::Resized(logical_size) => {
                                println!("OldSize: {:?} New Size: {:?}", size, logical_size);
                                needs_resize = true;
                                size.width = logical_size.width;
                                size.height = logical_size.height;
                            },
                            _ => {},
                        }
                    },
                    Event::DeviceEvent { event, ..} => {
                        match event {
                            DeviceEvent::MouseMotion { delta } => {
                                new_delta = (delta.0 as f32, delta.1 as f32);
                            },
                            _ => {},
                        }
                    },
                    Event::EventsCleared => {

                        if needs_resize {
                            let specs_world = specs_world.read().unwrap();
                            let mut scene = specs_world.write_resource::<Scene<Backend>>();
                            
                            if let Some(graph) = graph.take() {
                                graph.dispose(&mut factory, &scene);
                            }

                            let new_surface = factory.create_surface(&window).expect("failed to create surface");

                            graph = build_graph(size, new_surface, &scene, &mut factory, &mut families);

                            needs_resize = false;
                        }

                        let specs_world = specs_world.read().unwrap();
                        
                        specs_world.write_resource::<InputState>().mouse_delta = new_delta;
                        new_delta = (0., 0.);

                        *specs_world.write_resource::<DeltaTime>() = DeltaTime(delta_time.elapsed().as_secs_f32());
                        delta_time = Instant::now();

                        rhai_engine.eval_with_scope::<()>(&mut scope, "dispatch_world(dispatcher, world)").expect("Failed to run rhai code");

                        let mut scene = specs_world.write_resource::<Scene<Backend>>();

                        factory.maintain(&mut families);
                        if let Some(ref mut graph) = graph {
                            graph.run(&mut factory, &mut families, &scene);
                        }
                    },
                    _ => {}
                }

                if *control_flow == ControlFlow::Exit {
                    let specs_world = specs_world.read().unwrap();
                    let mut scene = specs_world.write_resource::<Scene<Backend>>();
                    if let Some(graph) = graph.take() {
                        graph.dispose(&mut factory, &scene);
                    }
                }
            });
        }
    );
}

#[cfg(not(any(feature = "dx12", feature = "metal", feature = "vulkan")))]
fn main() {
    panic!("Specify feature: { dx12, metal, vulkan }");
}

fn degrees_to_radians<T>( deg: T) -> T 
where T: num::Float {
    deg * num::cast(0.0174533).unwrap()
}

// TODO: Create to convert GLTF Animation data to a fast-readable format (bake matrices at each bone)