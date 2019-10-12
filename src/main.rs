#![cfg_attr(
    not(any(feature = "dx12", feature = "metal", feature = "vulkan")),
    allow(unused)
)]

use std::sync::{Arc, Mutex};
use std::path::PathBuf;

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
    shader::{ShaderKind, SourceLanguage, PathBufShaderInfo, ShaderSet},
    wsi::winit::{EventsLoop, WindowBuilder, dpi},
};

extern crate nalgebra_glm as glm;

extern crate specs;
use specs::prelude::*;
#[macro_use]
extern crate specs_derive;

extern crate rhai;
use rhai::{RegisterFn, Scope};

use rendy::shader::SpirvReflection;

#[cfg(feature = "dx12")]
type Backend = rendy::dx12::Backend;

#[cfg(feature = "metal")]
type Backend = rendy::metal::Backend;

#[cfg(feature = "vulkan")]
type Backend = rendy::vulkan::Backend;

lazy_static::lazy_static! {
    static ref VERTEX: PathBufShaderInfo = PathBufShaderInfo::new(
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/data/shaders/triangle.vert")),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    );

    static ref FRAGMENT: PathBufShaderInfo = PathBufShaderInfo::new(
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/data/shaders/triangle.frag")),
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

struct Scene<B: hal::Backend> {
    test_model : GltfModel,
    phantom: std::marker::PhantomData<B>,
}

#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
struct UniformData {
    model_matrix : glm::Mat4,
    view_matrix  : glm::Mat4,
    proj_matrix  : glm::Mat4,
}

#[derive(Debug, Default)]
struct TriangleRenderPipelineDesc;

#[derive(Debug)]
struct TriangleRenderPipeline<B: hal::Backend> {
    vertex: Option<Escape<Buffer<B>>>,
    sets: Vec<Escape<DescriptorSet<B>>>,
    uniform_buffer: Escape<Buffer<B>>
}

impl<B> SimpleGraphicsPipelineDesc<B, Scene<B>> for TriangleRenderPipelineDesc
where
    B: hal::Backend,
{
    type Pipeline = TriangleRenderPipeline<B>;

    fn depth_stencil(&self) -> Option<hal::pso::DepthStencilDesc> {
        None
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
        _scene : &Scene<B>,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<TriangleRenderPipeline<B>, failure::Error> {
        assert!(buffers.is_empty());
        assert!(images.is_empty());
        assert!(set_layouts.len() > 0);

        let frames_in_flight_count = ctx.frames_in_flight as u64;

        let uniform_size = std::mem::size_of::<UniformData>() as u64;
        let uniform_buffer = factory
            .create_buffer(
                BufferInfo {
                    size: uniform_size * frames_in_flight_count,
                    usage: hal::buffer::Usage::UNIFORM,
                },
                Dynamic,
            )
            .unwrap();

        let mut sets = Vec::new();
        for index in 0..frames_in_flight_count {
            sets.push( unsafe {
                let set = factory
                    .create_descriptor_set(set_layouts[0].clone())
                    .unwrap();
                factory.write_descriptor_sets(Some(hal::pso::DescriptorSetWrite {
                    set: set.raw(),
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(hal::pso::Descriptor::Buffer(
                        uniform_buffer.raw(),
                        Some(uniform_size * index) .. Some(uniform_size * (index + 1))
                    )),
                }));
                set
            });
        }

        Ok(TriangleRenderPipeline { vertex: None, sets, uniform_buffer})
    }
}

static mut ROTATION : f32 = 0.0f32;

impl<B> SimpleGraphicsPipeline<B, Scene<B>> for TriangleRenderPipeline<B>
where
    B: hal::Backend,
{
    type Desc = TriangleRenderPipelineDesc;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        index: usize,
        _scene : &Scene<B>,
    ) -> PrepareResult {
        if self.vertex.is_none() {
            let vbuf_size = SHADER_REFLECTION.attributes_range(..).unwrap().stride as u64 * 3;

            let mut vbuf = factory
                .create_buffer(
                    BufferInfo {
                        size: vbuf_size,
                        usage: hal::buffer::Usage::VERTEX,
                    },
                    Dynamic,
                )
                .unwrap();

            unsafe {
            // Fresh buffer.
            factory
                .upload_visible_buffer(
                    &mut vbuf,
                    0,
                    &[
                        PosColor {
                            position: [0.0, -0.5, 0.0].into(),
                            color: [1.0, 0.0, 0.0, 1.0].into(),
                        },
                        PosColor {
                            position: [0.5, 0.5, 0.0].into(),
                            color: [0.0, 1.0, 0.0, 1.0].into(),
                        },
                        PosColor {
                            position: [-0.5, 0.5, 0.0].into(),
                            color: [0.0, 0.0, 1.0, 1.0].into(),
                        },
                    ],
                ).unwrap();
            }

            self.vertex = Some(vbuf);
        }

        //Update Uniform Data (offset based on current frame in flight index)
        unsafe {
            ROTATION = ROTATION + 0.01f32;
            factory.upload_visible_buffer(
                &mut self.uniform_buffer,
                (std::mem::size_of::<UniformData>() * index) as u64, //access correct uni buffer
                &[UniformData {
                    model_matrix : glm::rotate_z(&glm::Mat4::identity(), ROTATION),
                    view_matrix  : glm::Mat4::identity(),
                    proj_matrix  : glm::Mat4::identity(),
                }]
            ).unwrap();
        }

        PrepareResult::DrawReuse
    }

    //FCS Note: this is only called once per frame-in-flight, prepare called every frame
    fn draw(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        _scene : &Scene<B>,
    ) {
        println!("Index: {}", index);
        unsafe {
            encoder.bind_graphics_descriptor_sets(
                layout,
                0,
                Some(self.sets[index].raw()),
                std::iter::empty(),
            );
            let vbuf = self.vertex.as_ref().unwrap();
            encoder.bind_vertex_buffers(0, Some((vbuf.raw(), 0)));
            encoder.draw(0..3, 0..1);
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _scene : &Scene<B>) {}
}

mod specs_systems;
use specs_systems::{common::DeltaTime, spatial::{Position, Velocity, Acceleration, VelocitySystem, AccelerationSystem, PrinterSystem}};

mod gltf_loader;
use gltf_loader::*;

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn main() {
    env_logger::Builder::from_default_env()
        .filter_module("triangle", log::LevelFilter::Trace)
        .init();

    let gltf_model = GltfModel::new("data/models/Running.glb");

    let mut scene = Scene {
        test_model : gltf_model,
        phantom : std::marker::PhantomData,
    };

    let config: Config = Default::default();

    let (mut factory, mut families): (Factory<Backend>, _) = rendy::factory::init(config).unwrap();

    let mut event_loop = EventsLoop::new();

    let window = WindowBuilder::new()
        .with_title("Rendy example")
        .with_dimensions(dpi::LogicalSize::new(1280.0 as f64, 720.0 as f64))
        .build(&event_loop)
        .unwrap();

    event_loop.poll_events(|_| ());

    let surface = factory.create_surface(&window);

    let mut graph_builder = GraphBuilder::<Backend, Scene<Backend>>::new();

    let size = window
        .get_inner_size()
        .unwrap()
        .to_physical(window.get_hidpi_factor());

    let color = graph_builder.create_image(
        hal::image::Kind::D2(size.width as u32, size.height as u32, 1, 1),
        1,
        factory.get_surface_format(&surface),
        Some(hal::command::ClearValue::Color([1.0, 1.0, 1.0, 1.0].into())),
    );

    let pass = graph_builder.add_node(
        TriangleRenderPipeline::builder()
            .into_subpass()
            .with_color(color)
            .into_pass(),
    );

    let present_builder = PresentNode::builder(&factory, surface, color).with_dependency(pass);

    let frames_in_flight_count = present_builder.image_count();

    graph_builder.add_node(present_builder);

    let mut graph = graph_builder
        .with_frames_in_flight(frames_in_flight_count)
        .build(&mut factory, &mut families, &mut scene)
        .unwrap();

    //Specs setup
    let specs_world = Arc::new(Mutex::new(World::new()));
    
    let specs_dispatcher = {
        let mut dispatcher = DispatcherBuilder::new()
            .with(VelocitySystem, "velocity_system", &[])
            .with(AccelerationSystem, "acceleration_system", &["velocity_system"])
            .with(PrinterSystem, "printer_system", &[])
        .build();
        //NOTE: has to be done before creating entities in world (seems problematic)
        dispatcher.setup(&mut specs_world.lock().unwrap());
        Arc::new(Mutex::new(dispatcher))
    };

    {
        let mut specs_world = specs_world.lock().unwrap();
        specs_world.insert(DeltaTime(0.05)); //add a resource
        specs_world.create_entity().with(Velocity(glm::vec3(2.0, 1.0, 3.0))).with(Position(glm::Vec3::new(1.0, 2.0, 3.0))).build();
        specs_world.create_entity().with(Velocity(glm::vec3(200.0, 100.0, 300.0))).with(Position(glm::Vec3::new(1.0, 2.0, 3.0))).build();
        
        specs_world.create_entity()
            .with(Velocity(glm::vec3(2.0, 1.0, 3.0)))
            .with(Position(glm::Vec3::new(1.0, 2.0, 3.0)))
            .with(Acceleration(glm::Vec3::new(2.0, 1.0, 2.0)))
            .build();

        specs_world.create_entity().with(Position(glm::Vec3::new(1.0, 2.0, 3.0))).build();
    }

    fn add(x : i64, y : i64) -> i64 {
        x + y
    }

    fn dispatch_world(dispatcher: Arc<Mutex<specs::Dispatcher>>, world : Arc<Mutex<specs::World>>) {
        dispatcher.lock().unwrap().dispatch_par(&world.lock().unwrap());
    }

    let mut rhai_engine = rhai::Engine::new();
    rhai_engine.register_fn("add", add);
    rhai_engine.register_fn("dispatch_world", dispatch_world);
    rhai_engine.register_type::<Arc<Mutex<specs::Dispatcher>>>();
    rhai_engine.register_type::<Arc<Mutex<specs::World>>>();
   
    let mut scope = Scope::new();
    scope.push(("dispatcher".to_string(), Box::new(specs_dispatcher.clone())));
    scope.push(("world". to_string(), Box::new(specs_world.clone())));

    loop {

        //TODO: Actually compute delta time
        *specs_world.lock().unwrap().write_resource::<DeltaTime>() = DeltaTime(0.04);

        //Execute Rhai Script
        rhai_engine.eval_with_scope::<()>(&mut scope, "dispatch_world(dispatcher, world)").expect("Failed to run rhai code");

        let mut should_close = false;
        event_loop.poll_events(|event| {
            match event {
                rendy::wsi::winit::Event::WindowEvent{window_id : _, event} => {
                    match event {
                        rendy::wsi::winit::WindowEvent::CloseRequested => {
                            should_close = true;
                        },
                        _ => {},
                    }
                },
                _ => {},
            }
        });

        if should_close { break; }
        
        //TODO: Wrap in render system
        factory.maintain(&mut families);
        event_loop.poll_events(|_| ());
        graph.run(&mut factory, &mut families, &mut scene);
    }

    graph.dispose(&mut factory, &mut scene);
}

#[cfg(not(any(feature = "dx12", feature = "metal", feature = "vulkan")))]
fn main() {
    panic!("Specify feature: { dx12, metal, vulkan }");
}

// TODO: 1. Crate to convert GLTF Animation data to a fast-readable format (bake matrices at each bone)
// TODO: 2. Create a RenderComponent
// TODO: 3. Render System at end that renders output