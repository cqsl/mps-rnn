#!/usr/bin/env julia

using HDF5
using ITensors
using MKL
using Random

# 2D Heisenberg
function main(;
    L::Int,
    max_B::Int,
    L2::Int = 0,
    peri::Bool = false,
    mars::Bool = true,
    J2::Number = 0,
    J22::Number = 0,
    J3::Number = 0,
    cmpl::Bool = false,
    zero_mag::Bool = true,
    snake::Bool = true,
    max_step::Int = 100,
    seed::Int = 0,
)
    @assert L % 2 == 0

    out_filename = "L$L"
    if L2 == 0
        L2 = L
    else
        @assert L2 % 2 == 0
        out_filename *= ",$L2"
    end
    if !peri
        out_filename *= "_open"
    end
    if mars
        out_filename *= "_mars"
    end
    if J2 != 0
        out_filename *= "_J2=$J2"
    end
    if J22 != 0
        out_filename *= "_J22=$J22"
    end
    if J3 != 0
        out_filename *= "_J3=$J3"
    end
    out_filename *= "_B$max_B"
    if cmpl
        out_filename *= "_cmpl"
    end
    if zero_mag
        out_filename *= "_zm"
    end
    if !snake
        out_filename *= "_ro_none"
    end
    out_filename *= ".hdf5"
    @show out_filename

    if seed > 0
        Random.seed!(seed)
    end

    sites = siteinds("S=1/2", L * L2; conserve_qns = zero_mag)

    ampo = OpSum()

    function ind(i, j)
        i = mod1(i, L)
        j = mod1(j, L2)
        if snake
            if i % 2 == 1
                return (i - 1) * L2 + j
            else
                return (i - 1) * L2 + (L2 + 1 - j)
            end
        else
            return (i - 1) * L2 + j
        end
    end

    function add_edge!(J, i1, j1, i2, j2)
        Jxy = 0.5 * J
        if mars
            d = i2 - i1 + j2 - j1
            if mod(d, 2) == 1
                Jxy *= -1
            end
        end

        k1 = ind(i1, j1)
        k2 = ind(i2, j2)
        ampo += J, "Sz", k1, "Sz", k2
        ampo += Jxy, "S+", k1, "S-", k2
        ampo += Jxy, "S-", k1, "S+", k2
    end

    for i = 1:L
        for j = 1:L2-1+peri
            add_edge!(1, i, j, i, j + 1)
        end
    end
    for i = 1:L-1+peri
        for j = 1:L2
            add_edge!(1, i, j, i + 1, j)
        end
    end

    # Diagonal term, needed for triangular lattice and J1-J2 model
    if J2 != 0
        for i = 1:L-1+peri
            for j = 1:L2-1+peri
                add_edge!(J2, i, j, i + 1, j + 1)
            end
        end
    end

    # Inverse diagonal term, needed for J1-J2 model
    if J22 != 0
        for i = 1:L-1+peri
            for j = 1:L2-1+peri
                add_edge!(J22, i, j + 1, i + 1, j)
            end
        end
    end

    # J3 term
    if J3 != 0
        for i = 1:L
            for j = 1:L2-2+peri*2
                add_edge!(J3, i, j, i, j + 2)
            end
        end
        for i = 1:L-2+peri*2
            for j = 1:L2
                add_edge!(J3, i, j, i + 2, j)
            end
        end
    end

    H = MPO(ampo, sites)

    dtype = cmpl ? ComplexF64 : Float64
    psi = nothing
    if zero_mag
        L_half = div(L, 2)
        L2_half = div(L2, 2)
        state = ["Up", "Dn"]
        state = repeat(state, L2_half)
        if snake
            state = vcat(state, reverse(state))
        else
            state = repeat(state, 2)
        end
        state = repeat(state, L_half)
        psi = randomMPS(dtype, sites, state; linkdims = max_B)
    else
        psi = randomMPS(dtype, sites; linkdims = max_B)
    end

    sweeps = Sweeps(max_step)
    setmaxdim!(sweeps, max_B)
    setcutoff!(sweeps, 1e-12)
    noise = 10 .^ LinRange(-3, -12, div(max_step, 2))
    setnoise!(sweeps, noise..., 0)

    energy, psi = dmrg(H, psi, sweeps)
    @show energy

    h5open(out_filename, "w") do f
        write(f, "psi", dense(psi))
    end

    GC.gc()

    H_sqr = real(inner(H, psi, H, psi))
    energy_std = sqrt(H_sqr - energy^2)
    @show energy_std

    return sites, H, psi
end
