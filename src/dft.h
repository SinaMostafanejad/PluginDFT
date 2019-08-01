/*
 * @BEGIN LICENSE
 *
 * mydft by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2016 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#ifndef DFT_SOLVER_H
#define DFT_SOLVER_H

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include "psi4/libmints/wavefunction.h"

// for dft
#include "psi4/libfock/v.h"
#include "psi4/libfunctional/superfunctional.h"

// for grid
#include "psi4/libfock/points.h"
#include "psi4/libfock/cubature.h"

namespace psi{ namespace mydft{

class DFTSolver: public Wavefunction{

  public:

    DFTSolver(std::shared_ptr<psi::Wavefunction> reference_wavefunction,Options & options);
    ~DFTSolver();
    void common_init();
    double compute_energy();
    virtual bool same_a_b_orbs() const { return same_a_b_orbs_; }
    virtual bool same_a_b_dens() const { return same_a_b_dens_; }

  protected:

   // //************************************************
   // //*      Constants for building functionals      *
   // //************************************************
   // // const double alpha_ = 0.773;
   // // const double alpha_ = 0.75;
   // const double alpha_ = (2.0/3.0);      // Gaspar, Kohn and Sham value
   // // const double alpha_ = 1.0;         // Slater's value
   // const double Cx_ = (9.0/8.0) * alpha_ * pow(3.0/M_PI,1.0/3.0);

   // static const double pa_ = 1.0;

   // static const double Aa_ = 0.0168869;
   // static const double a1a_ = 0.11125;
   // static const double b1a_ = 10.357;
   // static const double b2a_ = 3.6231;
   // static const double b3a_ = 0.88026;
   // static const double b4a_ = 0.49671;

   // static const double pe_ = 1.0;

   // static const double c0p_ = 0.0310907;
   // static const double a1p_ = 0.21370;
   // static const double b1p_ = 7.5957;
   // static const double b2p_ = 3.5876;
   // static const double b3p_ = 1.6382;
   // static const double b4p_ = 0.49294;

   // static const double c0f_ = 0.01554535;
   // static const double a1f_ = 0.20548;
   // static const double b1f_ = 14.1189;
   // static const double b2f_ = 6.1977;
   // static const double b3f_ = 3.3662;
   // static const double b4f_ = 0.62517;

   // static const double BETA_ = 0.06672455060314922;
   // static const double MU_ = 0.2195149727645171;
   // static const double KAPPA_ = 0.804;
   // static const double GAMMA_ = 0.0310906908696549;

   // /// Value of the second derivative of f(zeta) weight factor at zeta = 0 
   // static const double d2fZet0_ = 1.709921;

    /// the nuclear repulsion energy
    double enuc_;

    /// evaluate the orbital gradient
    std::shared_ptr<Matrix> OrbitalGradient(std::shared_ptr<Matrix> D,
                                            std::shared_ptr<Matrix> F,
                                            std::shared_ptr<Matrix> Shalf);

    /// xc potential matrices
    std::shared_ptr<Matrix> Va_;
    std::shared_ptr<Matrix> Vb_;

    /// explicitly evaluate exchange correlation energy 
    double ExchangeCorrelationEnergy();

    /// level of differentiation 
    int deriv_;

    /// is gga?
    bool is_gga_;

    /// is meta?
    bool is_meta_;

    /// is unpolarized (restricted)?
    bool is_unpolarized_;

    /// number of grid points_
    long int phi_points_;

    /// phi matrix
    std::shared_ptr<Matrix> super_phi_;

    /// d phi / dx matrix
    std::shared_ptr<Matrix> super_phi_x_;

    /// d phi / dy matrix
    std::shared_ptr<Matrix> super_phi_y_;

    /// d phi / dz matrix
    std::shared_ptr<Matrix> super_phi_z_;

    /// grid x values
    std::shared_ptr<Vector> grid_x_;

    /// grid y values
    std::shared_ptr<Vector> grid_y_;

    /// grid z values
    std::shared_ptr<Vector> grid_z_;

    /// grid weights
    std::shared_ptr<Vector> grid_w_;

    /// alpha spin density
    std::shared_ptr<Vector> rho_a_;

    /// beta spin density
    std::shared_ptr<Vector> rho_b_;

    /// total spin density
    std::shared_ptr<Vector> rho_;

    /// alpha spin density gradient x
    std::shared_ptr<Vector> rho_a_x_;

    /// beta spin density gradient x
    std::shared_ptr<Vector> rho_b_x_;

    /// alpha spin density gradient y
    std::shared_ptr<Vector> rho_a_y_;

    /// beta spin density gradient y
    std::shared_ptr<Vector> rho_b_y_;

    /// alpha spin density gradient z
    std::shared_ptr<Vector> rho_a_z_;

    /// beta spin density gradient z
    std::shared_ptr<Vector> rho_b_z_;

    /// inner product of alpha density gradient with itself (gamma_aa)
    std::shared_ptr<Vector> sigma_aa_;
    
    /// inner product of beta density gradient with itself (gamma_bb)
    std::shared_ptr<Vector> sigma_bb_;

    /// inner product of alpha density gradient with beta density gradient (gamma_ab)
    std::shared_ptr<Vector> sigma_ab_;

    /// total density gradient 
    std::shared_ptr<Vector> sigma_;
    
    /// derivative of the functional with respect to rho_a_
    std::shared_ptr<Vector> vrho_a_;

    /// derivative of the functional with respect to rho_b_
    std::shared_ptr<Vector> vrho_b_;

    /// derivative of the functional with respect to gamma_aa_
    std::shared_ptr<Vector> vsigma_aa_;

    /// derivative of the functional with respect to gamma_bb_
    std::shared_ptr<Vector> vsigma_bb_;

    /// derivative of the functional with respect to gamma_ab_
    std::shared_ptr<Vector> vsigma_ab_;

    /// 2nd derivative of the functional with respect to alpha density
    std::shared_ptr<Vector> v2rho_a2_;

    /// 2nd derivative of the functional with respect to alpha and beta densities
    std::shared_ptr<Vector> v2rho_ab_;

    /// 2nd derivative of the functional with respect to beta density
    std::shared_ptr<Vector> v2rho_b2_;

    /// 2nd derivative of the functional with respect to rho_a_ and gamma_aa_
    std::shared_ptr<Vector> v2rho_a_sigma_aa_;

    /// 2nd derivative of the functional with respect to rho_b_ and gamma_bb_
    std::shared_ptr<Vector> v2rho_b_sigma_bb_;

    /// 2nd derivative of the functional with respect to gamma_aa_
    std::shared_ptr<Vector> v2sigma_aa2_;

    /// 2nd derivative of the functional with respect to gamma_bb_
    std::shared_ptr<Vector> v2sigma_bb2_;

    /// 2nd derivative of the functional with respect to rho_a_ and gamma_ab_ 
    std::shared_ptr<Vector> v2rho_a_sigma_ab_;

    /// 2nd derivative of the functional with respect to rho_b_ and gamma_ab_ 
    std::shared_ptr<Vector> v2rho_b_sigma_ab_;

    /// 2nd derivative of the functional with respect to rho_a_ and gamma_bb_ 
    std::shared_ptr<Vector> v2rho_a_sigma_bb_;

    /// 2nd derivative of the functional with respect to rho_b_ and gamma_aa_ 
    std::shared_ptr<Vector> v2rho_b_sigma_aa_;

    /// 2nd derivative of the functional with respect to gamma_aa_ and gamma_ab_ 
    std::shared_ptr<Vector> v2sigma_aa_ab_;

    /// 2nd derivative of the functional with respect to gamma_aa_ and gamma_bb_ 
    std::shared_ptr<Vector> v2sigma_aa_bb_;

    /// 2nd derivative of the functional with respect to gamma_ab_ 
    std::shared_ptr<Vector> v2sigma_ab2_;

    /// 2nd derivative of the functional with respect to gamma_ab_ and gamma_bb_ 
    std::shared_ptr<Vector> v2sigma_ab_bb_;

    /// a function to build phi/phi_x/...
    void BuildPhiMatrix(std::shared_ptr<VBase> potential, std::shared_ptr<PointFunctions> points_func,
            std::string phi_type, std::shared_ptr<Matrix> myphi);

    /// S^{-1/2}
    std::shared_ptr<Matrix> Shalf_;
    std::shared_ptr<Matrix> Shalf2;

    /// build G spin-interpolation formula
    double Gfunction(double r, double A, double a1, double b1, double b2, double b3, double b4, double p);

    /// zeta factor zeta = ( rho_a(r) - rho_b(r) ) / ( rho_a(r) + rho_b(r) )
    std::shared_ptr<Vector> zeta_;

    /// spin magnetization factor / net spin-density  m = rho_a(r) - rho_b(r)
    std::shared_ptr<Vector> m_;

    /// effective radius of density
    std::shared_ptr<Vector> rs_;

    /// tau kinetic energy of alpha electrons
    std::shared_ptr<Vector> tau_a_;
    
    /// tau kinetic energy of beta electrons
    std::shared_ptr<Vector> tau_b_;

    /// build spin densities and gradients
    void BuildRho(std::shared_ptr<Matrix> Da, std::shared_ptr<Matrix> Db);

    //############################################################
    //############ Exchange functions' declarations ##############
    //############################################################

    /// build EX_LDA(rho)
    double EX_LDA(std::shared_ptr<Vector> rho_a, std::shared_ptr<Vector> rho_b);

    /// build EX_LSDA(rho_sigma)
    double EX_LSDA_Sigma(std::shared_ptr<Vector> rho_sigma);

    /// build EX_LSDA(rho_a, rho_b)
    double EX_LSDA_I(std::shared_ptr<Vector> rho_a, std::shared_ptr<Vector> rho_b);
    
    /// build EX_LSDA(rho, zeta)
    double EX_LSDA_III(std::shared_ptr<Vector> rho, std::shared_ptr<Vector> zeta);
    
    /// build EX_B86_MGC()
    double EX_B86_MGC();

    /// build EX_B88()
    double EX_B88();
    double EX_B88_I();

    /// build EX_B3()
    double EX_B3();

    /// build EX_PBE()
    double EX_PBE();
    double EX_PBE_I();
    double EX_wPBE_I();

    /// build EX_RPBE()
    double EX_RPBE();

    /// build EX_UPBE()
    double EX_UPBE();

    //############################################################
    //########### Correlation functions' declarations ############
    //############################################################

    /// build EC_B88()
    double EC_B88_OP();
   
    /// build EC_LYP()
    double EC_LYP();
    double EC_LYP_I();

    /// build EC_PBE()
    double EC_PBE();
    double EC_PBE_I();
    
    /// build EC_VWN3_RPA()
    double EC_VWN3_RPA();
    double EC_VWN3_RPA_III();
    double EC_VWN5_I();
    double EC_VWN5_RPA_I();
     
    //#####################################################################
    //########### Exchange-correlation functions' declarations ############
    //#####################################################################

    /// build EXC_B3LYP()
    double EXC_B3LYP();

};

}}


#endif

