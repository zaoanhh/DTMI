using Pkg
Pkg.add("CairoMakie")
Pkg.add("LinearAlgebra")
Pkg.add("Random")
Pkg.add("Statistics")
using CairoMakie, LinearAlgebra, Random, Statistics

function music(; sig, f, c, delta_d, num_tx)
    N, K = size(sig)# N:接收天线数，K:信号长度
    ang2rad = π / 180
    d = range(0, step=delta_d, length=N)
    X1 = sig
    Rx = Hermitian(X1 * X1' / K)
    M = num_tx
    F = eigen(Rx)
    V = F.vectors  # 特征向量构成的矩阵
    # 对特征向量矩阵进行列反转
    EV = V[:, end:-1:1]
    En = EV[:, (M+1):end]  # 噪声子空间
    p_music = zeros(361)
    angles_all = zeros(Float64, 361)
    for i in 1:361
        angle_i = (i - 181) / 2
        angles_all[i] = angle_i
        theta_m = angle_i * ang2rad
        a = conj(exp.(-im * 2π * (f / c) .* sin.(theta_m) .* d))

        p_music[i] = real(1 / (a' * En * En' * a))
    end
    p_max = maximum(p_music)
    p_music_db = 10 .* log10.(p_music ./ p_max)  # 归一化处理至分贝
    return (angles_all, p_music_db)
end

function awgn(signal::AbstractArray{T}, snr_db::Real) where {T <: Number}
    snr_linear = 10^(snr_db / 10)
    signal_power = mean(abs2.(signal))
    noise_variance = signal_power / snr_linear
    noise = sqrt(noise_variance) * randn(T, size(signal)...)
    return signal + noise
end

N = 8
M = 2
theta = [-30, 60]
snr = 10
K = 512
dd = 0.03
d = range(0, step=dd, length=N)
f = 5e9
c = 3e8
ang2rad = π / 180
A = exp.(-im * 2π * (f / c) .* transpose(d) .* sin.(theta .* ang2rad))'

S = rand(M, K)
X = A * S
@info "$(size(transpose(d)))"
X1 = awgn(X, 10)
angle_all, p_music = music(sig=X1, f=f, c=c, delta_d=dd, num_tx=M)
display(lines(angle_all, p_music))
angle_all[argmax(p_music)]