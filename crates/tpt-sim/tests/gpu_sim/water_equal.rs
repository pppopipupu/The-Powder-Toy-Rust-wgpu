use super::*;

#[test]
fn test_water_equalisation_u_tube() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(200, 150);
    engine.bufs.config.water_equal_test = true;

    // Build U-tube:
    // Left outer wall: x in 30..=33, y in 40..=120
    fill_rect(&mut engine, 30, 40, 33, 120, PT_DMND);
    // Middle divider wall: x in 45..=48, y in 40..=105
    fill_rect(&mut engine, 45, 40, 48, 105, PT_DMND);
    // Right outer wall: x in 60..=63, y in 40..=120
    fill_rect(&mut engine, 60, 40, 63, 120, PT_DMND);
    // Bottom wall: x in 30..=63, y in 117..=120
    fill_rect(&mut engine, 30, 117, 63, 120, PT_DMND);

    // Left arm interior: x in 34..=44, y in 40..=116
    // Bottom channel interior: x in 34..=59, y in 106..=116
    // Right arm interior: x in 49..=59, y in 40..=105

    // Fill water in left arm and bottom channel:
    fill_rect(&mut engine, 34, 50, 44, 116, PT_WATR);
    fill_rect(&mut engine, 45, 106, 59, 116, PT_WATR);
    engine.bufs.sync_particles_gpu_to_host();
    let count_right_arm = |eng: &SimulationEngine| -> usize {
        let mut count = 0;
        for i in 0..eng.bufs.capacity {
            if eng.bufs.part_ty[i] == PT_WATR {
                let x = eng.bufs.part_x[i];
                let y = eng.bufs.part_y[i];
                if (49.0..=59.0).contains(&x) && y <= 105.0 {
                    count += 1;
                }
            }
        }
        count
    };

    let initial = count_right_arm(&engine);
    assert_eq!(initial, 0, "right arm must start empty above row 105");

    // Run ~120 ticks:
    for _ in 0..120 {
        step(&mut engine);
    }

    let final_water = count_right_arm(&engine);
    assert!(
        final_water > 10,
        "water must rise into the right arm of the U-tube, found {final_water} particles"
    );
}
