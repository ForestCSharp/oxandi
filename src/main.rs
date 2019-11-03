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

macro_rules! map(
    { $($key:expr => $value:expr),+ } => {
        {
            let mut m = ::std::collections::HashMap::new();
            $(
                m.insert($key, $value);
            )+
            m
        }
     };
);

#[macro_use]
extern crate derive_deref;

use rendy::{
    command::{QueueId, RenderPassEncoder},
    factory::{Config, Factory},
    graph::{
        present::PresentNode, render::*, GraphBuilder, GraphContext, NodeBuffer, NodeImage,
    },
    hal::{self, Device as _},
    memory::Dynamic,
    mesh::PosColor,
    resource::{Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle},
    shader::{ShaderKind, SourceLanguage, PathBufShaderInfo, ShaderSet, SpirvReflection},
    wsi::winit::{EventsLoop, WindowBuilder, dpi, VirtualKeyCode},
    wsi::winit as winit,
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

lazy_static::lazy_static! {
    static ref VERTEX: PathBufShaderInfo = PathBufShaderInfo::new(
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/data/shaders/basic.vert")),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    );  

    static ref FRAGMENT: PathBufShaderInfo = PathBufShaderInfo::new(
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/data/shaders/basic.frag")),
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    );

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).unwrap()
        .with_fragment(&*FRAGMENT).unwrap();
}

lazy_static::lazy_static! {
    static ref SHADER_REFLECTION: SpirvReflection = SHADERS.reflect().unwrap();
}

static WIDTH : f64 = 1280.0;
static HEIGHT : f64 = 720.0;

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
struct BasicRenderPipelineDesc;

impl<B> SimpleGraphicsPipelineDesc<B, Scene<B>> for BasicRenderPipelineDesc
where
    B: hal::Backend,
{
    type Pipeline = BasicRenderPipeline<B>;

    fn depth_stencil(&self) -> Option<hal::pso::DepthStencilDesc> {
        Some(hal::pso::DepthStencilDesc {
            depth: Some(hal::pso::DepthTest {
                    fun: hal::pso::Comparison::GreaterEqual,
                    write: true,
            }),
            depth_bounds: false,
            stencil: None,
        })
    }

    fn load_shader_set(
        &self, 
        factory: &mut Factory<B>, 
        _scene : &Scene<B>
    ) -> ShaderSet<B> {
        SHADERS.build(factory, Default::default()).unwrap()
    }

    fn vertices( &self ) -> Vec<(
        Vec<hal::pso::Element<hal::format::Format>>,
        hal::pso::ElemStride,
        hal::pso::VertexInputRate,
    )> {
        vec![SHADER_REFLECTION
            .attributes_range(..)
            .unwrap()
            .gfx_vertex_input_desc(hal::pso::VertexInputRate::Vertex)]
    }

    fn layout(&self) -> Layout {
        SHADER_REFLECTION.layout().unwrap()
    }

    fn build<'a>(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        _queue: QueueId,
        scene : &Scene<B>,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<BasicRenderPipeline<B>, failure::Error> {
        Ok(BasicRenderPipeline{
            mesh_data : HashMap::new(),
        })
    }
}

static mut ROTATION : f32 = 0.0f32;

#[derive(Debug, Default)]
struct BasicRenderPipelineMeshData<B: hal::Backend> {
    pub descriptor_sets : Vec<Escape<DescriptorSet<B>>>, //1 per frame in flight
    pub mvp_uniform_buffers : Vec<Escape<Buffer<B>>>, //1 per frame in flight
}

#[derive(Debug)]
struct BasicRenderPipeline<B: hal::Backend> {
    mesh_data : HashMap<u32, BasicRenderPipelineMeshData<B>>,
}

impl<B> SimpleGraphicsPipeline<B, Scene<B>> for BasicRenderPipeline<B>
where
    B: hal::Backend,
{
    type Desc = BasicRenderPipelineDesc;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
        index: usize,
        scene : &Scene<B>,
    ) -> PrepareResult {
        assert!(set_layouts.len() > 0);
        
        //TODO: destroy mesh_data for meshes that no longer exist

        let specs_world = scene.specs_world.read().expect("Failed to read from scene.specs_world RwLock");
        let specs_entities = specs_world.entities();
        let specs_meshes = specs_world.read_storage::<MeshComponent<B>>();
        let specs_transforms = specs_world.read_storage::<Transform>();

        for (entity, mesh, transform) in (&specs_entities, &specs_meshes, &specs_transforms).join() {
            let id = entity.id();
            if !self.mesh_data.contains_key(&id) {
                let uniform_size = std::mem::size_of::<UniformData>() as u64;

                let mut uniform_buffers = Vec::with_capacity(scene.frames_in_flight as usize);
                let mut sets = Vec::with_capacity(scene.frames_in_flight as usize);

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

                    sets.push( unsafe {
                        let set = factory
                            .create_descriptor_set(set_layouts[0].clone())
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
                        set
                    });     
                }

                self.mesh_data.insert(id, BasicRenderPipelineMeshData {
                    descriptor_sets : sets,
                    mvp_uniform_buffers : uniform_buffers,
                });
            }

            if let Some(mut mesh_data) = self.mesh_data.get_mut(&entity.id()) {
                unsafe {
                    ROTATION = ROTATION + 0.01f32;

                    let model_matrix = {
                        let mut model_matrix = glm::identity();
                        model_matrix = glm::translate(&model_matrix, &transform.position);
                        model_matrix *= glm::quat_to_mat4(&transform.rotation);
                        //TODO: Scale
                        model_matrix
                    };

                    let mut perspective_matrix = glm::perspective_zo(
                        WIDTH as f32 / HEIGHT as f32,
                        degrees_to_radians(60.0f32),
                        100000.0,
                        0.01
                    );
                    perspective_matrix[(1,1)] *= -1.0;

                    factory.upload_visible_buffer(
                        &mut mesh_data.mvp_uniform_buffers[index], //access correct uni buffer
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
                }
            }
        }

        /* Alternatively, DrawReuse only records command buffers once, useful when rendered items never changes 
            (i.e. fullscreen quad for GBuff lighting / PostProcess) */
        PrepareResult::DrawRecord
    }

    fn draw(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        scene : &Scene<B>,
    ) {
        unsafe {
            let specs_world = scene.specs_world.read().expect("Failed to read from scene.specs_world RwLock");
            let specs_entities = specs_world.entities();
            let specs_mesh_storage = specs_world.read_storage::<MeshComponent<B>>();
            for (entity, mesh) in (&specs_entities, &specs_mesh_storage).join() {
                if let Some(mesh_data) = self.mesh_data.get(&entity.id()) {
                    encoder.bind_graphics_descriptor_sets(
                        layout,
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

    fn dispose(self, _factory: &mut Factory<B>, _scene : &Scene<B>) {}
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

    let config: Config = Default::default();

    let (mut factory, mut families): (Factory<Backend>, _) = rendy::factory::init(config).unwrap();

    let mut event_loop = EventsLoop::new();

    let window = WindowBuilder::new()
        .with_title("Oxandi")
        .with_dimensions(dpi::LogicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)
        .unwrap();

    event_loop.poll_events(|_| ());

    let surface = factory.create_surface(&window);

    let mut graph_builder = GraphBuilder::<Backend, Scene<Backend>>::new();

    let size = window
        .get_inner_size()
        .unwrap()
        .to_physical(window.get_hidpi_factor());

    let window_kind = hal::image::Kind::D2(size.width as u32, size.height as u32, 1, 1);

    let color = graph_builder.create_image(
        window_kind,
        1,
        factory.get_surface_format(&surface),
        Some(hal::command::ClearValue::Color([1.0, 1.0, 1.0, 1.0].into())),
    );

    let depth = graph_builder.create_image(
        window_kind,
        1,
        hal::format::Format::D16Unorm,
        Some(hal::command::ClearValue::DepthStencil(
            hal::command::ClearDepthStencil(0.0, 0),
        )),
    );

    let basic_pass = graph_builder.add_node(
        BasicRenderPipeline::builder()
            .into_subpass()
            .with_color(color)
            .with_depth_stencil(depth)
            .into_pass(),
    );

    let present_pass = PresentNode::builder(&factory, surface, color).with_dependency(basic_pass);

    let frames_in_flight_count = present_pass.image_count();

    graph_builder.add_node(present_pass);

    //Specs setup
    let specs_world = Arc::new(RwLock::new(World::new()));

    let mut scene = Scene {
        specs_world : specs_world.clone(),
        camera_data : CameraRenderData::new(),
        frames_in_flight : frames_in_flight_count,
        phantom : std::marker::PhantomData,
    };

    let mut graph = graph_builder
        .with_frames_in_flight(frames_in_flight_count)
        .build(&mut factory, &mut families, &mut scene)
        .unwrap();
    
    let specs_dispatcher = {
        let mut dispatcher = DispatcherBuilder::new()
            .with(InputMovementSystem, "input_movement_system", &[])
            .with(UpdateVelocitySystem, "acceleration_system", &[])
            .with(UpdatePositionSystem, "update_position_system", &["input_movement_system", "acceleration_system"])
            .with(UpdateRotationSystem, "update_rotation_system", &["input_movement_system"])
            .with(UpdateCameraSystem, "update_camera_system", &["input_movement_system"])
        .build();
        //NOTE: has to be done before creating entities in world (seems problematic)
        dispatcher.setup(&mut specs_world.write().unwrap());
        Arc::new(RwLock::new(dispatcher))
    };

    {
        let mut specs_world = specs_world.write().unwrap();
        specs_world.insert(DeltaTime(0.05)); //add a resource
        specs_world.insert(InputState::new());
        specs_world.insert(scene);
        specs_world.create_entity().with(Velocity(glm::vec3(2.0, 1.0, 3.0))).with(Transform::new()).build();
        specs_world.create_entity().with(Velocity(glm::vec3(200.0, 100.0, 300.0))).with(Transform::new()).build();

        specs_world.register::<MeshComponent<Backend>>(); //FIXME: can remove this once a system using MeshComponent is hooked up to world
        specs_world.create_entity().with(Transform::new()).with(MeshComponent::new(&mut factory)).build();
        
        specs_world.create_entity()
            .with(MeshComponent::new(&mut factory))
            .with(Transform::new())
            .with(Velocity(glm::vec3(0.5, 0.0, 0.0)))
            .with(AngularVelocity(glm::quat(0.0, 1.0, 1.0, 5.0)))
            .build();
        
        //Camera
        specs_world.create_entity()
            .with(Transform::new())
            .with(Velocity(glm::vec3(0.0, 0.0, 0.0)))
            .with(Acceleration(glm::vec3(0.0, 0.0, 0.0)))
            .with(Drag(10.0))
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

    loop {

        //Execute Rhai Script
        rhai_engine.eval_with_scope::<()>(&mut scope, "dispatch_world(dispatcher, world)").expect("Failed to run rhai code");

        let mut should_close = false;
        let mut new_delta = (0., 0.);
        event_loop.poll_events(|event| {
            match event {
                winit::Event::WindowEvent{window_id : _, event} => {
                    match event {
                        winit::WindowEvent::CloseRequested => {
                            should_close = true;
                        },
                        winit::WindowEvent::KeyboardInput { device_id : _, input } => {
                            match input.virtual_keycode {
                                Some(keycode) => {
                                    let specs_world = specs_world.read().unwrap();
                                    specs_world.write_resource::<InputState>().key_states.insert(keycode, input.state == winit::ElementState::Pressed);

                                    match keycode {
                                        VirtualKeyCode::Escape => should_close = true,
                                        _ => {},
                                    }
                                },
                                _ => {},
                            }
                        },
                        winit::WindowEvent::MouseInput { state, button, ..} => {
                            let pressed = state == winit::ElementState::Pressed;
                            let specs_world = specs_world.read().unwrap();
                            let mouse_button_states = &mut specs_world.write_resource::<InputState>().mouse_button_states;
                            match button {
                                winit::MouseButton::Left   => mouse_button_states[0] = pressed,
                                winit::MouseButton::Right  => mouse_button_states[1] = pressed,
                                winit::MouseButton::Middle => mouse_button_states[2] = pressed,
                                winit::MouseButton::Other(idx) => {
                                    if (idx as usize) < mouse_button_states.len() {
                                        mouse_button_states[idx as usize] = pressed;
                                    }
                                }
                            }
                        },
                        _ => {},
                    }
                },
                winit::Event::DeviceEvent { event, ..} => {
                    match event {
                        winit::DeviceEvent::MouseMotion { delta } => {
                            new_delta = (delta.0 as f32, delta.1 as f32);
                        },
                        _ => {},
                    }
                }
                _ => {},
            }
        });

        {
            let specs_world = specs_world.read().unwrap();
            *specs_world.write_resource::<DeltaTime>() = DeltaTime(delta_time.elapsed().as_secs_f32());
            delta_time = Instant::now();
            specs_world.write_resource::<InputState>().mouse_delta = new_delta;
        }

        if should_close { break; }
        
        //TODO: Wrap in render system
        factory.maintain(&mut families);
        event_loop.poll_events(|_| ());
        {
            let specs_world = specs_world.read().unwrap();
            let mut scene = specs_world.write_resource::<Scene<Backend>>();
            graph.run(&mut factory, &mut families, &mut scene);
        }
    }

    {
        let specs_world = specs_world.read().unwrap();
        let mut scene = specs_world.write_resource::<Scene<Backend>>();
        graph.dispose(&mut factory, &mut scene);
        //TODO: cleanup mesh vertex/index buffers?
    }
}

#[cfg(not(any(feature = "dx12", feature = "metal", feature = "vulkan")))]
fn main() {
    panic!("Specify feature: { dx12, metal, vulkan }");
}

fn degrees_to_radians<T>( deg: T) -> T 
where T: num::Float {
    deg * num::cast(0.0174533).unwrap()
}

// TODO: 1. Crate to convert GLTF Animation data to a fast-readable format (bake matrices at each bone)
// TODO: 2. Create a RenderComponent
// TODO: 3. Render System at end that renders output