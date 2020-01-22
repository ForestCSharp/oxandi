use specs::prelude::*;

use rendy::{
    hal,
    hal::{device::Device},
    factory::{Config, Factory},
    memory::Dynamic,
    resource::{Escape, Buffer, BufferInfo, DescriptorSetLayout, Handle},
    shader::{ShaderKind, SourceLanguage, PathBufShaderInfo, ShaderSet, SpirvReflection},
};

use std::path::PathBuf;

use crate::gltf_loader::*;
use crate::DeltaTime;

#[derive(Component)]
pub struct MeshComponent<B: hal::Backend> {
    pub vertex_buffer: Escape<Buffer<B>>,
    pub index_buffer : Escape<Buffer<B>>,
    pub num_indices  : u32,
    pub gltf_model : GltfModel,
}

impl<B> MeshComponent<B> where B: hal::Backend {
    pub fn new(mesh_path : &str, factory: &mut Factory<B>) -> MeshComponent<B> {
        let gltf_model = GltfModel::new(mesh_path);
        let vertices = &gltf_model.meshes[0].vertices;
        let indices = &gltf_model.meshes[0].indices.as_ref().unwrap();

        let vbuf_size = std::mem::size_of::<Vertex>() as u64 * vertices.len() as u64;

        let mut vbuf = factory
            .create_buffer(
                BufferInfo {
                    size: vbuf_size,
                    usage: hal::buffer::Usage::VERTEX,
                },
                Dynamic,
            ).unwrap();

        unsafe {
        // Fresh buffer.
        factory
            .upload_visible_buffer(
                &mut vbuf,
                0,
                &vertices,
            ).unwrap();
        }

        let ibuf_size = (std::mem::size_of::<usize>() * indices.len()) as u64;

        let mut ibuf = factory
            .create_buffer(
                BufferInfo {
                    size  : ibuf_size,
                    usage : hal::buffer::Usage::INDEX,
                },
                Dynamic,
            ).unwrap();

        unsafe {
        factory
            .upload_visible_buffer(
                &mut ibuf,
                0,
                &indices,
            ).unwrap();
        }

        MeshComponent {
            vertex_buffer : vbuf,
            index_buffer  : ibuf,
            num_indices   : indices.len() as u32,
            gltf_model,
        }
    }
}

//TODO: make material component just reference existing materials? (to save resources)

//Material for main color pass
#[derive(Component)]
pub struct Material<B : hal::Backend> {
    pub shader_set : ShaderSet<B>,
    pub set_layouts : Vec<Handle<DescriptorSetLayout<B>>>,
    pub pipeline_layout : B::PipelineLayout,
    pub vertex_buffer_descs : Vec<hal::pso::VertexBufferDesc>,
    pub attribute_descs : Vec<hal::pso::AttributeDesc>,
}

impl<B> Material<B> where B : hal::Backend {
    pub fn new( vertex_shader_path : &str, fragment_shader_path : &str, factory: &mut Factory<B>) -> Material<B> {

        let manifest_dir = env!("CARGO_MANIFEST_DIR");

        //FIXME: add leading slashes if not present in paths

        let vertex_shader = PathBufShaderInfo::new(
            PathBuf::from(manifest_dir.to_owned() + vertex_shader_path),
            ShaderKind::Vertex,
            SourceLanguage::GLSL,
            "main",
        );

        let fragment_shader = PathBufShaderInfo::new(
            PathBuf::from(manifest_dir.to_owned() + fragment_shader_path),
            ShaderKind::Fragment,
            SourceLanguage::GLSL,
            "main",
        );

        let shader_set_builder = rendy::shader::ShaderSetBuilder::default()
            .with_vertex(&vertex_shader).unwrap()
            .with_fragment(&fragment_shader).unwrap();

        let shader_reflection = shader_set_builder.reflect().map_err(|e| println!("{}", e)).unwrap();

        let shader_set = shader_set_builder.build(factory, Default::default()).unwrap();

        let set_layouts = shader_reflection
            .layout().unwrap()
            .sets
            .into_iter()
            .map(|set| {
                factory
                    .create_descriptor_set_layout(set.bindings)
                    .map(Handle::from)
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to create descriptor set layouts");

        let pipeline_layout = unsafe {
            factory.device().create_pipeline_layout(
                set_layouts.iter().map(|l| l.raw()),
                &[],
            ).expect("failed to create pipeline layout")
        };

        let vertex_layout = vec![shader_reflection
            .attributes_range(..)
            .unwrap()
            .gfx_vertex_input_desc(hal::pso::VertexInputRate::Vertex)
        ];

        let mut vertex_buffer_descs = Vec::new();
        let mut attribute_descs = Vec::new();

        for &(ref elements, stride, rate) in &vertex_layout {
            let index = vertex_buffer_descs.len() as hal::pso::BufferIndex;

            vertex_buffer_descs.push(hal::pso::VertexBufferDesc {
                binding: index,
                stride,
                rate,
            });
        
            let mut location = attribute_descs.last().map_or(0, |a : &hal::pso::AttributeDesc| a.location + 1);
            for &element in elements {
                attribute_descs.push(hal::pso::AttributeDesc {
                    location,
                    binding: index,
                    element,
                });
                location += 1;
            }
        }

        Material {
            shader_set : shader_set,
            set_layouts,
            pipeline_layout,
            vertex_buffer_descs,
            attribute_descs,
        }
    }

    pub fn create_pipeline(&self, subpass : hal::pass::Subpass<B>, factory : &Factory<B>) -> B::GraphicsPipeline {
        unsafe {
            factory.device().create_graphics_pipeline(
                &hal::pso::GraphicsPipelineDesc {
                    shaders : self.shader_set.raw().expect("failed to get raw shader set"),
                    rasterizer: hal::pso::Rasterizer { //TODO: params for this
                        polygon_mode : hal::pso::PolygonMode::Fill,
                        cull_face : hal::pso::Face::NONE,
                        front_face : hal::pso::FrontFace::CounterClockwise,
                        depth_clamping : false,
                        depth_bias : None,
                        conservative : false,
                    },
                    vertex_buffers : self.vertex_buffer_descs.clone(),
                    attributes : self.attribute_descs.clone(),
                    input_assembler: hal::pso::InputAssemblerDesc {
                        primitive: hal::pso::Primitive::TriangleList,
                        with_adjacency: false,
                        restart_index: None,
                    },
                    blender: hal::pso::BlendDesc {
                        logic_op: None,
                        targets: vec![hal::pso::ColorBlendDesc { //TODO: base this length off of render group's color attachments
                            mask: hal::pso::ColorMask::ALL,
                            blend: Some(hal::pso::BlendState::ALPHA),
                        }],
                    },
                    depth_stencil: hal::pso::DepthStencilDesc {
                        depth: Some(hal::pso::DepthTest {
                                fun: hal::pso::Comparison::GreaterEqual,
                                write: true,
                            }),
                        depth_bounds: false,
                        stencil: None,
                    },
                    multisampling: None,
                    baked_states: hal::pso::BakedStates::default(), //FIXME: need to now set scissor and viewport via cmd buffer
                    layout: &self.pipeline_layout,
                    subpass,
                    flags: hal::pso::PipelineCreationFlags::empty(),
                    parent: hal::pso::BasePipeline::None,
                },
                None,
            ).expect("failed to create graphics pipeline")
        }
    }
}
#[derive(Default, Component)]
pub struct AnimComponent
{
    pub speed : f32,
}

impl AnimComponent {
    pub fn new(speed : f32) -> AnimComponent {
        AnimComponent {
            speed,
        }
    }
}

pub struct AnimationSystem<B : hal::Backend>
{
    pub phantom : std::marker::PhantomData<B>,
}

impl<'a, B> System<'a> for AnimationSystem<B>
    where B : hal::Backend 
    {

    type SystemData = ( Read<'a, DeltaTime>,
                        WriteStorage<'a, MeshComponent<B>>,
                        ReadStorage<'a, AnimComponent>,
                      );

     fn run(&mut self, (dt, mut meshes, animations): Self::SystemData) {
         for (mesh, anim) in (&mut meshes, &animations).join() {
            //TODO: skeleton index, anim index
            mesh.gltf_model.animate(0, 0, (anim.speed * dt.0) as f64);
         }
     }
}