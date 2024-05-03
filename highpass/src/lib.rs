extern crate std;

/// single order highpass filter. 
/// https://en.wikipedia.org/wiki/High-pass_filter - has pseudo code 

pub fn highpass_filter(data: &mut [f32], sampling_rate: f32, cutoff_frequency: f32) {

    let rc = 1.0 / (cutoff_frequency * 2.0 * core::f32::consts::PI);
    let dt = 1.0 / sampling_rate;
    let alpha = rc / (rc + dt);

    data[0] *= 1.0; // y[1] := x[1]
    for i in 1..data.len() {
        // data is accessed before being overwritten
        data[i] = alpha * (data[i - 1] + data[i] - data[i - 1])
    }
}
