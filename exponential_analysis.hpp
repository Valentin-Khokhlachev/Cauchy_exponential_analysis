#ifndef EXPONENTIAL_ANALYSIS_HPP
#define EXPONENTIAL_ANALYSIS_HPP

#include <filesystem>
#include <omp.h>

#include "mptypes.hpp"

struct global_params
{
    unsigned int N = 0;

    mpt::complex_t t_s{"0.0", "0.0"};
    mpt::complex_t t_f{"0.0", "0.0"};
    mpt::complex_t h_t{"0.0", "0.0"};

    mpt::complex_t z_0{"0.0", "0.0"};
    mpt::complex_t R{"0.0", "0.0"};

    mpt::complex_t s_s{"0.0", "0.0"};
    mpt::complex_t s_f{"0.0", "0.0"};
    mpt::complex_t h_s{"0.0", "0.0"};
};

struct func_params
{
    mpt::complex_t p1{"0.0", "0.0"};
    mpt::complex_t p2{"0.0", "0.0"};
};

struct real_mesh_and_weights
{
    mpt::complex_vector_t t{};
    mpt::complex_t h_t{"0.0", "0.0"};
    mpt::complex_vector_t g_t{};
};

struct complex_mesh_and_weights
{
    mpt::complex_vector_t z{};
    mpt::complex_t h_s{"0.0", "0.0"};
    mpt::complex_vector_t g_z{};
};

void test_libraries_call();
void test_omp_with_eigen();
bool check_folder(const std::string &path);

void init_global_params(global_params &gp);
void init_func_params(func_params &fp);
void init_real_mesh_linspace(const mpt::complex_t &start, const mpt::complex_t &finish, const unsigned int &num_of_points, real_mesh_and_weights &rm);

void init_complex_mesh(const global_params &gp, complex_mesh_and_weights &cm);
void init_vector_zeros(const global_params &gp, mpt::complex_vector_t &v);
void init_u_int(const real_mesh_and_weights &rm, const func_params &fp, mpt::complex_vector_t &u_int);
void init_u(const real_mesh_and_weights &rm, const func_params &fp, mpt::complex_vector_t &u);
void init_u_der(const real_mesh_and_weights &rm, const func_params &fp, mpt::complex_vector_t &u_der);

void ea_get_interp_coeffs(const real_mesh_and_weights &rm, const mpt::complex_vector_t &u, const complex_mesh_and_weights &cm, mpt::complex_vector_t &u_wave, const global_params &gp, const func_params &fp);
void ea_eval_interp(const complex_mesh_and_weights &cm, const mpt::complex_vector_t &u_wave, const real_mesh_and_weights &rm, mpt::complex_vector_t &u, const global_params &gp, const func_params &fp, const int &denom_power = 1);
void ea_eval_interp_int(const complex_mesh_and_weights &cm, const mpt::complex_vector_t &u_wave, const real_mesh_and_weights &rm, mpt::complex_vector_t &u_int, const global_params &gp, const func_params &fp);
mpt::complex_t norm_L2_real_mesh(const real_mesh_and_weights &rm, const mpt::complex_vector_t &u);
mpt::complex_t norm_diff_L2_real_mesh(const real_mesh_and_weights &rm, const mpt::complex_vector_t &u1, const mpt::complex_vector_t &u2);
void complex_vector_diff(const mpt::complex_vector_t &v1, const mpt::complex_vector_t &v2, mpt::complex_vector_t &res);

mpt::complex_t func_u_int(const mpt::complex_t &t, const func_params &fp);
mpt::complex_t func_u(const mpt::complex_t &t, const func_params &fp);
mpt::complex_t func_u_der(const mpt::complex_t &t, const func_params &fp);

#endif // EXPONENTIAL_ANALYSIS_HPP