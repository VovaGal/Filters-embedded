extern crate rulinalg;

use rulinalg::matrix::{BaseMatrix, Matrix};
use rulinalg::vector::Vector;

/// * `q`: process noise covariance
/// * `r`: measurement noise covariance
/// * `h`: observation matrix
/// * `f`: state transition matrix
/// * `x0`: initial guess for state mean at time 1
/// * `p0`: initial guess for state covariance at time 1
#[derive(Debug)]
pub struct KalmanFilter {
    pub q: Matrix<f64>,   // Process noise covariance
    pub r: Matrix<f64>,   // Measurement noise covariance
    pub h: Matrix<f64>,   // Observation matrix
    pub f: Matrix<f64>,   // State transition matrix
    pub x0: Vector<f64>,  // State variable initial value
    pub p0: Matrix<f64>   // State covariance initial value
}

#[derive(Clone, Debug)]
pub struct KalmanState {
    pub x: Vector<f64>,   // State vector
    pub p: Matrix<f64>    // State covariance
}

impl KalmanFilter {
    pub fn filter(&self, data: &Vec<Vector<f64>>) -> (Vec<KalmanState>, Vec<KalmanState>) {

        let t: usize = data.len();

        // Containers for predicted and filtered estimates
        let mut predicted: Vec<KalmanState> = Vec::with_capacity(t+1);
        let mut filtered: Vec<KalmanState> = Vec::with_capacity(t);

        predicted.push(KalmanState { x: (self.x0).clone(),
                                    p: (self.p0).clone() });

        for k in 0..t {
            filtered.push(update_step(self, &predicted[k], &data[k]));
            predicted.push(predict_step(self, &filtered[k]));
        }

        (filtered, predicted)
    }

    pub fn smooth(&self,
                filtered: &Vec<KalmanState>,
                predicted: &Vec<KalmanState>)
                -> Vec<KalmanState> {

        let t: usize = filtered.len();
        let mut smoothed: Vec<KalmanState> = Vec::with_capacity(t);

        // Do Kalman smoothing in reverse order
        let mut init = (filtered[t - 1]).clone();
        smoothed.push((filtered[t - 1]).clone());

        for k in 1..t {
            smoothed.push(smoothing_step(self, &init,
                                        &filtered[t-k-1],
                                        &predicted[t-k]));
            init = (&smoothed[k]).clone();
        }

        smoothed.reverse();
        smoothed
    }
}

pub fn predict_step(kalman_filter: &KalmanFilter,
                    init: &KalmanState)
                    -> KalmanState {

    // Predict state variable and covariance
    let xp: Vector<f64> = &kalman_filter.f * &init.x;
    let pp: Matrix<f64> = &kalman_filter.f * &init.p * &kalman_filter.f.transpose() +
        &kalman_filter.q;

    KalmanState { x: xp, p: pp}
}

pub fn update_step(kalman_filter: &KalmanFilter,
                pred: &KalmanState,
                measure: &Vector<f64>)
                -> KalmanState {

    let identity = Matrix::<f64>::identity(kalman_filter.x0.size());

    // Compute Kalman gain
    let k: Matrix<f64> = &pred.p * &kalman_filter.h.transpose() *
        (&kalman_filter.h * &pred.p * &kalman_filter.h.transpose() + &kalman_filter.r)
        .inverse()
        .expect("Kalman gain computation failed due to failure to invert.");

    // Update state variable and covariance
    let x = &pred.x + &k * (measure - &kalman_filter.h * &pred.x);
    let p = (identity - &k * &kalman_filter.h) * &pred.p;

    KalmanState { x: x, p: p }

}

pub fn filter_step(kalman_filter: &KalmanFilter,
                init: &KalmanState,
                measure: &Vector<f64>)
                -> (KalmanState, KalmanState) {

    let pred = predict_step(kalman_filter, init);
    let upd = update_step(kalman_filter, &pred, measure);

    (KalmanState { x: upd.x, p: upd.p }, KalmanState { x: pred.x, p: pred.p })
}


fn smoothing_step(kalman_filter: &KalmanFilter,
                init: &KalmanState,
                filtered: &KalmanState,
                predicted: &KalmanState)
                -> KalmanState {

    let j: Matrix<f64> = &filtered.p * &kalman_filter.f.transpose() *
        &predicted.p.clone().inverse()
        .expect("Predicted state covariance matrix could not be inverted.");
    let x: Vector<f64> = &filtered.x + &j * (&init.x - &predicted.x);
    let p: Matrix<f64> = &filtered.p + &j * (&init.p - &predicted.p) * &j.transpose();

    KalmanState { x: x, p: p }

}