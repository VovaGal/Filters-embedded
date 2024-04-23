use audio_visualizer::dynamic::live_input::AudioDevAndCfg;
use audio_visualizer::dynamic::window_top_btm::{open_window_connect_audio, TransformFn};
//use lowpass::lowpass_filter;
use highpass::highpass_filter;

/// -- release for smoother display
fn main() {
    open_window_connect_audio(
        "Filter View",
        None,
        None,
        None,
        None,
        "time (seconds)",
        "Amplitude (w filter)",
        // default audio input device
        AudioDevAndCfg::new(None, None),
        // lowpass filter
        TransformFn::Basic(|x, sampling_rate| {
            let mut data = x.iter().copied().collect::<Vec<_>>();
            highpass_filter(&mut data, sampling_rate, 30.0);
            data
        }),
    );
}