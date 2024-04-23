extern crate std;

/// single order low pass filter. Uses Simple infinite impulse response filter. Mutable buffer, single precisioun to the data provided.
/// The effect of an infinite impulse response low-pass filter can be simulated by analyzing
///an RC filter's behavior in the time domain, and then discretizing the model.
/// single for 32 bits to represent floating-point number, swap fot 64 for double (more memory, more prescision)
/// https://en.wikipedia.org/wiki/Low-pass_filter#Simple_infinite_impulse_response_filter - has pseudo code 

///single precision
pub fn lowpass_filter(data: &mut [f32], sampling_rate: f32, cutoff_frequency: f32) {
    // rc = 1/2*pi*fc
    let rc = 1.0 / (cutoff_frequency * 2.0 * core::f32::consts::PI);
    // sampling period = 1/ (1/T) or sampling f 
    let dt = 1.0 / sampling_rate;
    // smoothing factor = dt/rc + dt
    let alpha = dt / (rc + dt);

    data[0] *= alpha;
    for i in 1..data.len() {
        // data is accessed before being overwritten
        data[i] = data[i - 1] + alpha * (data[i] - data[i - 1]);
    }
}


