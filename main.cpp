#include <iostream>
#include <chrono>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>

#include "exponential_analysis.hpp"

int main()
{
    setlocale(LC_ALL, "");

    omp_set_num_threads(6);
    Eigen::setNbThreads(6);
    // std::cout << Eigen::nbThreads() << std::endl;

    // test_libraries_call();

    std::chrono::time_point<std::chrono::high_resolution_clock> time_start = std::chrono::high_resolution_clock::now();

    std::string path_to_res_folder = std::string(SOURCE_DIR) + std::string("/results/");
    std::string current_test = "interpolation/";
    check_folder(path_to_res_folder + current_test);
    std::string name_conv = "convergence.csv";
    std::string path_to_conv = path_to_res_folder + current_test + name_conv;
    std::string current_calc_folder = "";
    std::string name_init = "";
    std::string name_interp = "";

    std::stringstream ss;
    std::ofstream ofs_tmp;
    std::ofstream ofs_conv;
    ofs_conv.open(path_to_conv, std::ios::out);
    if (ofs_conv.is_open())
    {
        ofs_conv << "N;log10_abs_err_interp;log10_abs_err_der\n";

        for (unsigned int q = 3; q < 40; ++q)
        {
            ss << q;
            current_calc_folder = std::string("N=") + ss.str() + std::string("/");
            check_folder(path_to_res_folder + current_test + current_calc_folder);
            name_init = std::string("N=") + ss.str() + std::string("_init.csv");
            name_interp = std::string("N=") + ss.str() + std::string("_interp.csv");
            ss.str("");
            ss.clear();

            global_params gp;
            gp.N = q;
            init_global_params(gp);

            func_params fp;
            init_func_params(fp);

            real_mesh_and_weights rm;
            init_real_mesh_linspace(gp.t_s, gp.t_f, gp.N, rm);
            real_mesh_and_weights rm_interp;
            init_real_mesh_linspace(gp.t_s + (gp.h_t / mpt::complex_t("2.0", "0.0")), gp.t_f - (gp.h_t / mpt::complex_t("2.0", "0.0")), gp.N - 1, rm_interp);

            complex_mesh_and_weights cm;
            init_complex_mesh(gp, cm);

            mpt::complex_vector_t u{};
            init_u(rm, fp, u);
            mpt::complex_vector_t u_wave{};
            u_wave.assign(gp.N, mpt::complex_t("0.0", "0.0"));

            mpt::complex_vector_t u_int_exact{};
            init_u_int(rm_interp, fp, u_int_exact);
            mpt::complex_vector_t u_interp_exact{};
            init_u(rm_interp, fp, u_interp_exact);
            mpt::complex_vector_t u_der_exact{};
            init_u_der(rm_interp, fp, u_der_exact);

            mpt::complex_vector_t u_int{};
            mpt::complex_vector_t diff_int{};
            mpt::complex_vector_t u_interp{};
            mpt::complex_vector_t diff_interp{};
            mpt::complex_vector_t u_der{};
            mpt::complex_vector_t diff_der{};

            //---------------------------------------------------------------------------------------------------------
            //=========================================================================================================
            ea_get_interp_coeffs(rm, u, cm, u_wave, gp, fp);
            //=========================================================================================================
            ea_eval_interp_int(cm, u_wave, rm_interp, u_int, gp, fp);
            for (mpt::complex_vector_t::iterator it_u_int = u_int.begin(); it_u_int != u_int.end(); ++it_u_int)
            {
                (*it_u_int) += func_u_int((*rm_interp.t.cbegin()), fp);
            }
            complex_vector_diff(u_int_exact, u_int, diff_int);
            //=========================================================================================================
            ea_eval_interp(cm, u_wave, rm_interp, u_interp, gp, fp, 1);
            complex_vector_diff(u_interp_exact, u_interp, diff_interp);
            //=========================================================================================================
            ea_eval_interp(cm, u_wave, rm_interp, u_der, gp, fp, 2);
            complex_vector_diff(u_der_exact, u_der, diff_der);
            //=========================================================================================================
            //---------------------------------------------------------------------------------------------------------

            mpt::complex_t err_int = bm::log10(norm_diff_L2_real_mesh(rm_interp, u_int, u_int_exact));
            mpt::complex_t err_interp = bm::log10(norm_diff_L2_real_mesh(rm_interp, u_interp, u_interp_exact));
            mpt::complex_t err_der = bm::log10(norm_diff_L2_real_mesh(rm_interp, u_der, u_der_exact));

            ofs_conv << gp.N << ";"
                     << err_int.str(mpt::prc_out) << ";"
                     << err_interp.str(mpt::prc_out) << ";"
                     << err_der.str(mpt::prc_out) << "\n";

            //---------------------------------------------------------------------------------------------------------
            // output
            //---------------------------------------------------------------------------------------------------------
            if (!ofs_tmp.is_open())
            {
                ofs_tmp.open(path_to_res_folder + current_test + current_calc_folder + name_init, std::ios::out);
                if (ofs_tmp.is_open())
                {
                    ofs_tmp << "t;h_t;g_t;u;z;h_s;g_z;u_wave\n";

                    mpt::complex_vector_t::const_iterator it_g_t = rm.g_t.cbegin();
                    mpt::complex_vector_t::const_iterator it_u = u.cbegin();
                    mpt::complex_vector_t::const_iterator it_z = cm.z.cbegin();
                    mpt::complex_vector_t::const_iterator it_g_z = cm.g_z.cbegin();
                    mpt::complex_vector_t::const_iterator it_u_wave = u_wave.cbegin();

                    for (mpt::complex_vector_t::const_iterator it_t = rm.t.cbegin(); it_t != rm.t.cend(); ++it_t)
                    {
                        ofs_tmp << it_t->str(mpt::prc_out) << ";"
                                << rm.h_t.str(mpt::prc_out) << ";"
                                << it_g_t->str(mpt::prc_out) << ";"
                                << it_u->str(mpt::prc_out) << ";"
                                << it_z->str(mpt::prc_out) << ";"
                                << cm.h_s.str(mpt::prc_out) << ";"
                                << it_g_z->str(mpt::prc_out) << ";"
                                << it_u_wave->str(mpt::prc_out) << "\n";

                        ++it_g_t;
                        ++it_u;
                        ++it_z;
                        ++it_g_z;
                        ++it_u_wave;
                    }

                    ofs_tmp.close();
                }
            }
            //---------------------------------------------------------------------------------------------------------
            if (!ofs_tmp.is_open())
            {
                ofs_tmp.open(path_to_res_folder + current_test + current_calc_folder + name_interp, std::ios::out);
                if (ofs_tmp.is_open())
                {
                    ofs_tmp << "t_interp;h_t_interp;g_t_interp;u_interp;u_interp_exact;diff_interp;u_der;u_der_exact;diff_der\n";

                    mpt::complex_vector_t::const_iterator it_g_t_interp = rm_interp.g_t.cbegin();
                    mpt::complex_vector_t::const_iterator it_u_interp = u_interp.cbegin();
                    mpt::complex_vector_t::const_iterator it_u_interp_exact = u_interp_exact.cbegin();
                    mpt::complex_vector_t::const_iterator it_diff_interp = diff_interp.cbegin();
                    mpt::complex_vector_t::const_iterator it_u_der = u_der.cbegin();
                    mpt::complex_vector_t::const_iterator it_u_der_exact = u_der_exact.cbegin();
                    mpt::complex_vector_t::const_iterator it_diff_der = diff_der.cbegin();

                    for (mpt::complex_vector_t::const_iterator it_t_interp = rm_interp.t.cbegin(); it_t_interp != rm_interp.t.cend(); ++it_t_interp)
                    {
                        ofs_tmp << it_t_interp->str(mpt::prc_out) << ";"
                                << rm_interp.h_t.str(mpt::prc_out) << ";"
                                << it_g_t_interp->str(mpt::prc_out) << ";"
                                << it_u_interp->str(mpt::prc_out) << ";"
                                << it_u_interp_exact->str(mpt::prc_out) << ";"
                                << it_diff_interp->str(mpt::prc_out) << ";"
                                << it_u_der->str(mpt::prc_out) << ";"
                                << it_u_der_exact->str(mpt::prc_out) << ";"
                                << it_diff_der->str(mpt::prc_out) << "\n";

                        ++it_g_t_interp;
                        ++it_u_interp;
                        ++it_u_interp_exact;
                        ++it_diff_interp;
                        ++it_u_der;
                        ++it_u_der_exact;
                        ++it_diff_der;
                    }

                    ofs_tmp.close();
                }
            }
            //---------------------------------------------------------------------------------------------------------
        }

        ofs_conv.close();
    }
    else
    {
        std::cout << "Can't open file!" << std::endl;
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> time_finish = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken = " << (std::chrono::duration_cast<std::chrono::nanoseconds>(time_finish - time_start)).count() / 1e9 << "\n";

    return 0;
}