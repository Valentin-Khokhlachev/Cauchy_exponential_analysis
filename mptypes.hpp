#ifndef MPTYPES_HPP
#define MPTYPES_HPP

#include <vector>
#include <cmath>

#include <Eigen/Eigen>

#include <boost/config.hpp>
#include <boost/multiprecision/fwd.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_int/cpp_int_config.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_complex.hpp>
#include <boost/multiprecision/complex_adaptor.hpp>
#include <boost/multiprecision/number.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/operators.hpp>
#include <boost/multiprecision/eigen.hpp> /// Eigen спецификация для типов из boost::multiprecision

#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/mpc.hpp>

#include "settings.hpp"

namespace bm = boost::multiprecision;

/*!
 * \brief mpt - типы данных с произвольным числом значащих цифр
 * (из boost::multiprecision)
 */
namespace mpt
{
    static const unsigned int prc_calc = 128u;
    static const unsigned int prc_out = 16u;

    /*!
     * \brief mpFloat_t - тип с плавающей точкой и произвольной разрядностью
     */
    // typedef bm::number<bm::backends::gmp_float<prc_calc>> float_t;
    // typedef bm::number<bm::backends::cpp_dec_float<prc_calc>> float_t;

    /*!
     * \brief mpComplex_t - тип комплексного числа с плавающей точкой и произвольной разрядностью
     */
    typedef bm::number<bm::backends::mpc_complex_backend<prc_calc>, boost::multiprecision::et_on> complex_t;
    // typedef bm::mpc_complex_500 complex_t;
    // typedef bm::number<bm::complex_adaptor<bm::backends::cpp_dec_float<prc_calc>>> complex_t;

    typedef std::vector<complex_t> complex_vector_t;

    static const complex_t PI = bm::atan(mpt::complex_t("1.0", "0.0")) * mpt::complex_t("4.0", "0.0");
}

#endif // MPTYPES_HPP