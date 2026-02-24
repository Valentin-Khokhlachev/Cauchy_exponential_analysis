#include "exponential_analysis.hpp"

void test_libraries_call()
{
    mpt::complex_t tmp = bm::atan(mpt::complex_t("1.0", "0.0")) * mpt::complex_t("4.0", "0.0");
    std::cout << "\n===============================\n"
              << "Pi = "
              << tmp.str(mpt::prc_out)
              << "\n===============================\n";
}

void test_omp_with_eigen()
{
    setlocale(LC_ALL, "");

    omp_set_num_threads(6);
    Eigen::setNbThreads(6);
    std::cout << Eigen::nbThreads() << std::endl;

    // Размер матрицы
    const int n = 1024;

    // 1. Создаем случайную матрицу и вектор правой части
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);

    // 2. Замеряем время решения
    auto start = std::chrono::high_resolution_clock::now();

    Eigen::PartialPivLU<Eigen::MatrixXd> solver(A);
    Eigen::VectorXd x = solver.solve(b);

    auto end = std::chrono::high_resolution_clock::now();

    // 3. Выводим время
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << duration.count() / 1e3 << std::endl;

#pragma omp parallel
    {
        std::cout << "Hello from thread " << omp_get_thread_num() << std::endl;
    }
}

bool check_folder(const std::string &path)
{
    bool res = false;

    std::filesystem::path messy_path(path);
    std::filesystem::path normalized_path = std::filesystem::weakly_canonical(messy_path);

    if (!std::filesystem::exists(normalized_path))
    {
        std::filesystem::create_directories(normalized_path);
    }
    else
    {
        res = true;
    }

    return res;
}

void init_global_params(global_params &gp)
{
    gp.t_s = mpt::complex_t("-3.0", "0.0");
    gp.t_f = mpt::complex_t("3.0", "0.0");
    gp.h_t = (gp.t_f - gp.t_s) / (static_cast<mpt::complex_t>(gp.N) - mpt::complex_t("1.0", "0.0"));

    gp.z_0 = mpt::complex_t("0.0", "0.0");
    gp.R = mpt::complex_t("3.5", "0.0");

    gp.s_s = mpt::complex_t("0.0", "0.0");
    gp.s_f = mpt::complex_t("1.0", "0.0");
    gp.h_s = (gp.s_f - gp.s_s) / (static_cast<mpt::complex_t>(gp.N) - mpt::complex_t("1.0", "0.0"));
}

void init_func_params(func_params &fp)
{
    fp.p1 = mpt::complex_t{"0.0", "4.0"};
    fp.p2 = mpt::complex_t{"1.5", "0.0"};
}

void init_real_mesh_linspace(const mpt::complex_t &start, const mpt::complex_t &finish, const unsigned int &num_of_points, real_mesh_and_weights &rm)
{
    rm.t.resize(num_of_points);
    rm.h_t = (finish - start) / (static_cast<mpt::complex_t>(num_of_points) - 1);
    rm.g_t.resize(num_of_points);

    mpt::complex_vector_t::iterator it_rm_t = rm.t.begin();
    mpt::complex_vector_t::iterator it_rm_g_t = rm.g_t.begin();

    for (size_t counter = 0; counter < num_of_points; ++counter)
    {
        (*it_rm_t) = start + rm.h_t * counter;

        if ((counter == 0) || (counter == (num_of_points - 1)))
        {
            (*it_rm_g_t) = mpt::complex_t("0.5", "0.0");
        }
        else
        {
            (*it_rm_g_t) = mpt::complex_t("1.0", "0.0");
        }

        ++it_rm_t;
        ++it_rm_g_t;
    }
}

void init_complex_mesh(const global_params &gp, complex_mesh_and_weights &cm)
{
    cm.z.resize(gp.N);
    cm.h_s = gp.h_s;
    cm.g_z.resize(gp.N);

    mpt::complex_vector_t::iterator it_cm_z = cm.z.begin();
    mpt::complex_vector_t::iterator it_cm_g_z = cm.g_z.begin();

    for (size_t counter = 0; counter < gp.N; ++counter)
    {
        (*it_cm_z) = gp.z_0 + gp.R * bm::exp(mpt::complex_t("0.0", "2.0") * mpt::PI * (gp.s_s + gp.h_s * static_cast<mpt::complex_t>(counter)));

        if ((counter == 0) || (counter == (gp.N - 1)))
        {
            (*it_cm_g_z) = mpt::complex_t("0.0", "1.0") * mpt::PI * ((*it_cm_z) - gp.z_0);
        }
        else
        {
            (*it_cm_g_z) = mpt::complex_t("0.0", "2.0") * mpt::PI * ((*it_cm_z) - gp.z_0);
        }

        ++it_cm_z;
        ++it_cm_g_z;
    }
}

void init_vector_zeros(const global_params &gp, mpt::complex_vector_t &v)
{
    v.resize(gp.N);

    for (mpt::complex_vector_t::iterator it_v = v.begin(); it_v != v.end(); ++it_v)
    {
        (*it_v) = mpt::complex_t("0.0", "0.0");
    }
}

void init_u_int(const real_mesh_and_weights &rm, const func_params &fp, mpt::complex_vector_t &u_int)
{
    u_int.resize(rm.t.size());

    mpt::complex_vector_t::const_iterator it_t = rm.t.cbegin();

    for (mpt::complex_vector_t::iterator it_u_int = u_int.begin(); it_u_int != u_int.end(); ++it_u_int)
    {
        (*it_u_int) = func_u_int((*it_t), fp);
        ++it_t;
    }
}

void init_u(const real_mesh_and_weights &rm, const func_params &fp, mpt::complex_vector_t &u)
{
    u.resize(rm.t.size());

    mpt::complex_vector_t::const_iterator it_t = rm.t.cbegin();

    for (mpt::complex_vector_t::iterator it_u = u.begin(); it_u != u.end(); ++it_u)
    {
        (*it_u) = func_u((*it_t), fp);
        ++it_t;
    }
}

void init_u_der(const real_mesh_and_weights &rm, const func_params &fp, mpt::complex_vector_t &u_der)
{
    u_der.resize(rm.t.size());

    mpt::complex_vector_t::const_iterator it_t = rm.t.cbegin();

    for (mpt::complex_vector_t::iterator it_u_der = u_der.begin(); it_u_der != u_der.end(); ++it_u_der)
    {
        (*it_u_der) = func_u_der((*it_t), fp);
        ++it_t;
    }
}

void ea_get_interp_coeffs(const real_mesh_and_weights &rm, const mpt::complex_vector_t &u, const complex_mesh_and_weights &cm, mpt::complex_vector_t &u_wave, const global_params &gp, const func_params &fp)
{
    Eigen::MatrixX<mpt::complex_t> M;
    M.resize(gp.N, gp.N);

    unsigned int i = 0;
    unsigned int j = 0;
    mpt::complex_vector_t::const_iterator it_g_z = cm.g_z.cbegin();
    for (mpt::complex_vector_t::const_iterator it_t = rm.t.cbegin(); it_t != rm.t.cend(); ++it_t)
    {
        it_g_z = cm.g_z.cbegin();
        j = 0;
        for (mpt::complex_vector_t::const_iterator it_z = cm.z.cbegin(); it_z != cm.z.cend(); ++it_z)
        {
            M(i, j) = gp.h_s * (*it_g_z) / ((*it_z) - (*it_t));
            ++it_g_z;
            ++j;
        }
        ++i;
    }

    Eigen::VectorX<mpt::complex_t> u_wave_eig;
    u_wave_eig.resize(gp.N);
    Eigen::VectorX<mpt::complex_t> u_eig;
    u_eig.resize(gp.N);
    std::transform(u.begin(), u.end(), u_eig.data(),
                   [](const mpt::complex_t &a) -> mpt::complex_t
                   { return (mpt::complex_t("0.0", "2.0") * mpt::PI * a); });

    u_wave_eig = M.fullPivHouseholderQr().solve(u_eig);

    u_wave.resize(gp.N);
    std::copy(u_wave_eig.data(), u_wave_eig.data() + u_wave_eig.size(), u_wave.begin());
}

void ea_eval_interp(const complex_mesh_and_weights &cm, const mpt::complex_vector_t &u_wave, const real_mesh_and_weights &rm, mpt::complex_vector_t &u, const global_params &gp, const func_params &fp, const int &denom_power)
{
    u.resize(rm.t.size());

    mpt::complex_vector_t::const_iterator it_g_t = rm.g_t.cbegin();
    mpt::complex_vector_t::iterator it_u = u.begin();
    mpt::complex_vector_t::const_iterator it_g_z = cm.g_z.cbegin();
    mpt::complex_vector_t::const_iterator it_u_wave = u_wave.cbegin();

    mpt::complex_t tmp{"0.0", "0.0"};

    int counter = 0;

    for (mpt::complex_vector_t::const_iterator it_t = rm.t.cbegin(); it_t != rm.t.cend(); ++it_t)
    {
        it_g_z = cm.g_z.cbegin();
        it_u_wave = u_wave.cbegin();
        tmp = mpt::complex_t("0.0", "0.0");

        for (mpt::complex_vector_t::const_iterator it_z = cm.z.cbegin(); it_z != cm.z.cend(); ++it_z)
        {
            tmp += gp.h_s * (*it_g_z) * (*it_u_wave) / bm::pow(((*it_z) - (*it_t)), denom_power);
            ++it_g_z;
            ++it_u_wave;
        }

        (*it_u) = tmp / (mpt::complex_t("0.0", "2.0") * mpt::PI);

        ++it_g_t;
        ++it_u;
        ++counter;
    }
}

void ea_eval_interp_int(const complex_mesh_and_weights &cm, const mpt::complex_vector_t &u_wave, const real_mesh_and_weights &rm, mpt::complex_vector_t &u_int, const global_params &gp, const func_params &fp)
{
    u_int.resize(rm.t.size());

    mpt::complex_vector_t::const_iterator it_g_t = rm.g_t.cbegin();
    mpt::complex_vector_t::iterator it_u_int = u_int.begin();
    mpt::complex_vector_t::const_iterator it_g_z = cm.g_z.cbegin();
    mpt::complex_vector_t::const_iterator it_u_wave = u_wave.cbegin();

    mpt::complex_t tmp{"0.0", "0.0"};

    int counter = 0;

    for (mpt::complex_vector_t::const_iterator it_t = rm.t.cbegin(); it_t != rm.t.cend(); ++it_t)
    {
        it_g_z = cm.g_z.cbegin();
        it_u_wave = u_wave.cbegin();
        tmp = mpt::complex_t("0.0", "0.0");

        for (mpt::complex_vector_t::const_iterator it_z = cm.z.cbegin(); it_z != cm.z.cend(); ++it_z)
        {
            tmp += gp.h_s * (*it_g_z) * (*it_u_wave) * bm::log(((*it_z) - (*rm.t.cbegin())) / ((*it_z) - (*it_t)));
            ++it_g_z;
            ++it_u_wave;
        }

        (*it_u_int) = tmp / (mpt::complex_t("0.0", "2.0") * mpt::PI);

        ++it_g_t;
        ++it_u_int;
        ++counter;
    }
}

mpt::complex_t norm_L2_real_mesh(const real_mesh_and_weights &rm, const mpt::complex_vector_t &u)
{
    mpt::complex_t res("0.0", "0.0");
    mpt::complex_vector_t::const_iterator it_u = u.cbegin();

    int counter = 0;
    for (mpt::complex_vector_t::const_iterator it_g_t = rm.g_t.cbegin(); it_g_t != rm.g_t.cend(); ++it_g_t)
    {
        res += rm.h_t * (*it_g_t) * (*it_u) * (*it_u);
        ++it_u;
        ++counter;
    }
    res = bm::sqrt(bm::abs(res));

    return res;
}

mpt::complex_t norm_diff_L2_real_mesh(const real_mesh_and_weights &rm, const mpt::complex_vector_t &u1, const mpt::complex_vector_t &u2)
{
    mpt::complex_t res("0.0", "0.0");

    mpt::complex_vector_t diff_u{};
    diff_u.assign(u1.size(), mpt::complex_t("0.0", "0.0"));

    mpt::complex_vector_t::const_iterator it_u1 = u1.cbegin();
    mpt::complex_vector_t::const_iterator it_u2 = u2.cbegin();
    for (mpt::complex_vector_t::iterator it_diff_u = diff_u.begin(); it_diff_u != diff_u.end(); ++it_diff_u)
    {
        (*it_diff_u) = bm::abs((*it_u1) - (*it_u2));
        ++it_u1;
        ++it_u2;
    }

    res = norm_L2_real_mesh(rm, diff_u);
    return res;
}

void complex_vector_diff(const mpt::complex_vector_t &v1, const mpt::complex_vector_t &v2, mpt::complex_vector_t &res)
{
    res.resize(v1.size());

    mpt::complex_vector_t::iterator it_res = res.begin();
    mpt::complex_vector_t::const_iterator it_v2 = v2.cbegin();
    for (mpt::complex_vector_t::const_iterator it_v1 = v1.cbegin(); it_v1 != v1.cend(); ++it_v1)
    {
        (*it_res) = (*it_v1) - (*it_v2);
        ++it_v2;
        ++it_res;
    }
}

mpt::complex_t func_u_int(const mpt::complex_t &t, const func_params &fp)
{
    mpt::complex_t res{"0.0", "0.0"};
    res = (t * t - fp.p2 * fp.p2) / (t - fp.p1);
    return res;
}

mpt::complex_t func_u(const mpt::complex_t &t, const func_params &fp)
{
    mpt::complex_t res{"0.0", "0.0"};
    mpt::complex_t tmp = (t - fp.p1);
    res = (t * t - mpt::complex_t("2.0", "0.0") * fp.p1 * t + fp.p2 * fp.p2) / (tmp * tmp);
    return res;
}

mpt::complex_t func_u_der(const mpt::complex_t &t, const func_params &fp)
{
    mpt::complex_t res{"0.0", "0.0"};
    mpt::complex_t tmp = (t - fp.p1);
    res = mpt::complex_t("2.0", "0.0") * ((tmp * tmp) - (t * t - mpt::complex_t("2.0", "0.0") * fp.p1 * t + fp.p2 * fp.p2)) / (tmp * tmp * tmp);
    return res;
}
