#![cfg_attr(
    not(any(feature = "dx12", feature = "metal", feature = "vulkan")),
    allow(unused)
)]

use rendy::{
    command::{Families, QueueId, RenderPassEncoder},
    factory::{Config, Factory},
    graph::{
        present::PresentNode, render::*, Graph, GraphBuilder, GraphContext, NodeBuffer, NodeImage,
    },
    hal::{self, Device as _, PhysicalDevice as _},
    memory::Dynamic,
    mesh::PosColor,
    resource::{Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle},
    shader::{ShaderKind, SourceLanguage, StaticShaderInfo, ShaderSet},
    wsi::winit::{EventsLoop, WindowBuilder, dpi},
};

extern crate nalgebra_glm as glm;

extern crate specs;
use specs::prelude::*;
#[macro_use]
extern crate specs_derive;

use rendy::shader::SpirvReflection;

#[cfg(feature = "dx12")]
type Backend = rendy::dx12::Backend;

#[cfg(feature = "metal")]
type Backend = rendy::metal::Backend;

#[cfg(feature = "vulkan")]
type Backend = rendy::vulkan::Backend;

lazy_static::lazy_static! {
    static ref VERTEX: StaticShaderInfo = StaticShaderInfo::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/data/shaders/triangle.vert"),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    );

    static ref FRAGMENT: StaticShaderInfo = StaticShaderInfo::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/data/shaders/triangle.frag"),
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

#[derive(Debug)]
struct Scene<B: hal::Backend> {
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

// A component contains data which is associated with an entity.

#[derive(Debug, Component)]
struct Vel {
    x : f32,
    y : f32,
}

#[derive(Debug, Component)]
struct Pos {
    x: f32,
    y: f32,
}

struct MovementSystem;

impl<'a> System<'a> for MovementSystem {
    // These are the resources required for execution.
    // You can also define a struct and `#[derive(SystemData)]`,
    // see the `full` example.
    type SystemData = (WriteStorage<'a, Pos>, ReadStorage<'a, Vel>);

    fn run(&mut self, (mut pos, vel): Self::SystemData) {
        // The `.join()` combines multiple components,
        // so we only access those entities which have
        // both of them.
        // You could also use `par_join()` to get a rayon `ParallelIterator`.
        (&vel, &mut pos)
        .par_join()
        .for_each(|(vel, pos)| {
            pos.x += vel.x * 0.05;
            pos.y += vel.y * 0.05;

            //println!("{}, {}", pos.x, pos.y);
        });
    }
}

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn main() {
    env_logger::Builder::from_default_env()
        .filter_module("triangle", log::LevelFilter::Trace)
        .init();

    let mut scene = Scene { phantom : std::marker::PhantomData};

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
    let mut specs_world = World::new();
    let mut specs_dispatcher = DispatcherBuilder::new().with(MovementSystem, "sys_a", &[]).build();
    specs_dispatcher.setup(&mut specs_world);
    specs_world.create_entity().with(Vel { x: 2.0, y: 2.0}).with(Pos { x: 0.0, y: 0.0}).build();
    specs_world.create_entity().with(Vel{ x: 4.0, y: 4.0}).with(Pos { x: 0.0, y: 0.0}).build();
    specs_world.create_entity().with(Vel{ x: 4.0, y: 4.0}).with(Pos { x: 0.0, y: 0.0}).build();
    specs_world.create_entity().with(Pos { x: 0.0, y: 0.0}).build();

    loop {
        specs_dispatcher.dispatch_par(&specs_world);

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