use specs::prelude::*;

use rendy::init::winit::event::VirtualKeyCode;
use std::collections::HashMap;

use crate::specs_systems::spatial::{Transform, Velocity};

use super::super::Scene;
use super::super::Backend;

#[derive(Default)]
pub struct DeltaTime(pub f32); //TODO: Make f64?

#[derive(Default)]
pub struct InputState {
    pub key_states : HashMap<VirtualKeyCode,bool>,
    pub mouse_button_states : [bool; 5],
    pub mouse_delta : (f32, f32),
}

impl InputState {
    pub fn new() -> InputState {
        InputState {
            key_states : HashMap::new(),
            mouse_button_states : [false, false, false, false, false],
            mouse_delta : (0., 0.),
        }
    }
}

#[derive(Default, Component)]
pub struct InputComponent {
    pub move_speed : f32,
}

pub struct InputMovementSystem;

impl<'a> System<'a> for InputMovementSystem {

    type SystemData = ( Read<'a, DeltaTime>, 
                        Read<'a, InputState>, 
                        ReadStorage<'a, InputComponent>, 
                        WriteStorage<'a, Velocity>, 
                        WriteStorage<'a, Transform>,
                      );

    fn run(&mut self, (dt, input_state, input_component, mut velocity, mut transform): Self::SystemData) {
        for (input_component, velocity, mut transform) in (&input_component, &mut velocity, &mut transform).join() {
            let is_key_pressed = |keycode : VirtualKeyCode| {
                input_state.key_states.get(&keycode) == Some(&true)
            };
            
            //Normalize quaternion
            transform.rotation = glm::quat_normalize(&transform.rotation);

            let forward_vec = glm::quat_rotate_vec3(&transform.rotation, &glm::vec3(0.0, 0.0, 1.0));
            let up_vec      = glm::quat_rotate_vec3(&transform.rotation, &glm::vec3(0.0, 1.0, 0.0));
            let right_vec   = forward_vec.cross(&up_vec);

            fn degrees_to_radians<T>( deg: T) -> T 
            where T: num::Float {
                deg * num::cast(0.0174533).unwrap()
            }

            //Rotate Based on Mouse Delta
            if input_state.mouse_button_states[0] {
                let yaw_rotation = glm::quat_angle_axis(degrees_to_radians(-input_state.mouse_delta.0 * 100.0* dt.0) as f32, &glm::vec3(0., 1., 0.));
                let pitch_rotation = glm::quat_angle_axis(degrees_to_radians(-input_state.mouse_delta.1 * 100.0 * dt.0) as f32, &right_vec);
                transform.rotation = glm::quat_normalize(&(yaw_rotation * pitch_rotation * transform.rotation));
            }
            
            //TODO: move_speed variable in input component?
            let move_speed = input_component.move_speed;

            if is_key_pressed(VirtualKeyCode::W) {
                velocity.0 += forward_vec * dt.0 * move_speed;
            }

            if is_key_pressed(VirtualKeyCode::S) {
                velocity.0 -= forward_vec * dt.0 * move_speed;
            }

            if is_key_pressed(VirtualKeyCode::A) {
                velocity.0 -= right_vec * dt.0 * move_speed;
            }

            if is_key_pressed(VirtualKeyCode::D) {
                velocity.0 += right_vec * dt.0 * move_speed;
            }

            if is_key_pressed(VirtualKeyCode::Q) {
                velocity.0 -= up_vec * dt.0 * move_speed;
            }

            if is_key_pressed(VirtualKeyCode::E) {
                velocity.0 += up_vec * dt.0 * move_speed;
            }
        };
    }
}

#[derive(Default, Component)]
#[storage(NullStorage)]
pub struct ActiveCamera;

pub struct UpdateCameraSystem;

impl<'a> System<'a> for UpdateCameraSystem {

    type SystemData = ( Option<Write<'a, Scene<Backend>>>, ReadStorage<'a, ActiveCamera>, ReadStorage<'a, Transform>);

     fn run(&mut self, (mut scene, active_camera, transform): Self::SystemData) {
         
         let mut camera_already_updated = false;
         for (_, transform) in (&active_camera, &transform).join() {
            match &mut scene {
                Some(scene) => {
                    if camera_already_updated { println!("ERROR: Multiple Active Cameras found"); }

                    scene.camera_data.position = transform.position;
                    let forward_vec = glm::quat_rotate_vec3(&transform.rotation, &glm::vec3(0.0, 0.0, 1.0));
                    let up_vec      = glm::quat_rotate_vec3(&transform.rotation, &glm::vec3(0.0, 1.0, 0.0));
                    scene.camera_data.target = transform.position + forward_vec;
                    scene.camera_data.up     = up_vec;
                    camera_already_updated = true;
                }
                None => println!("Missing Rendy Scene Resource"),
            }
         }
     }
}