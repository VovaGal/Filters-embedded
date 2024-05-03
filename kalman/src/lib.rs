/// input xestimated, Pestimated, z, Q, R
pub fn kalman_filter(data: &mut [f32], sampling_rate: f32, cutoff_frequency: f32) {
    convert data to matrix
    initialise F and H matrix

    // predict state vector and covariance
    predict state vvector and covariance:
    xpredict = F * xestimated
    Ppredict = F * Pestimated * F^T + Q

    // estimnation
    S = H * Ppredict * H^T + R

    // compute kalman gain factor
    Kgain = Ppredict * H^T * S^-1

    // correction based on observation
    xupdated = xpredicted + Kgain * (z - H * xpredicted)
    Pupdated = Ppredicted  - Kgain * H * Ppredicted

    return xupdated, Pupdated
}
