import numpy as np

class Derivative_KARFS(object):

    def __init__(self, Grid, scale):

        print("loading class",__class__.__name__)

        self.NX = Grid[0]
        self.NY = Grid[1]
        self.NZ = Grid[2]

        self.scale = 1./scale

        self.dfdx  = np.zeros((self.NX, self.NY, self.NZ))
        self.dfdy  = np.zeros((self.NX, self.NY, self.NZ))

        self.ae =  4./  5.
        self.be = -1./  5.
        self.ce =  4./105.
        self.de = -1./280.

    def calc_dfdx(self, f):

        NX = self.NX

        for i in range(4):
            if i==0: self.dfdx[i,:,:] = (-11.*f[0,:,:] +18.*f[1,:,:] - 9.*f[2,:,:] + 2.*f[3,:,:])/ 6.*self.scale[0]
            if i==1: self.dfdx[i,:,:] = (- 2.*f[0,:,:] - 3.*f[1,:,:] + 6.*f[2,:,:] - 1.*f[3,:,:])/ 6.*self.scale[0]
            if i==2: self.dfdx[i,:,:] = (+ 1.*f[0,:,:] - 8.*f[1,:,:] - 1.*f[4,:,:] + 8.*f[3,:,:])/12.*self.scale[0]
            if i==3: self.dfdx[i,:,:] = (- 1.*f[0,:,:] + 9.*f[1,:,:] -45.*f[2,:,:] \
                                        + 1.*f[6,:,:] - 9.*f[5,:,:] + 45.*f[4,:,:])/60.*self.scale[0]

        for i in range(4, NX-4):
            self.dfdx[i,:,:] = (   self.de *( f[i+4,:,:] - f[i-4,:,:] ) \
                                 + self.ce *( f[i+3,:,:] - f[i-3,:,:] ) \
                                 + self.be *( f[i+2,:,:] - f[i-2,:,:] ) \
                                 + self.ae *( f[i+1,:,:] - f[i-1,:,:] ) ) * self.scale[0]
            
            #print('mk1', self.dfdx[i,:,:,])
            #print('mk2', 0.5*(f[i+1,:,:] - f[i-1,:,:])*self.scale[0])
            #self.dfdx[i,:,:] = 0.5*(f[i+1,:,:] - f[i-1,:,:])*self.scale[0]
    
        for i in range(NX-4, NX):
            if i==NX-1-0: self.dfdx[i,:,:] = -(-11.*f[NX-1,:,:] +18.*f[NX-2,:,:] - 9.*f[NX-3,:,:] + 2.*f[NX-4,:,:])/ 6.*self.scale[0]
            if i==NX-1-1: self.dfdx[i,:,:] = -(- 2.*f[NX-1,:,:] - 3.*f[NX-2,:,:] + 6.*f[NX-3,:,:] - 1.*f[NX-4,:,:])/ 6.*self.scale[0]
            if i==NX-1-2: self.dfdx[i,:,:] = -(+ 1.*f[NX-1,:,:] - 8.*f[NX-2,:,:] - 1.*f[NX-5,:,:] + 8.*f[NX-4,:,:])/12.*self.scale[0]
            if i==NX-1-3: self.dfdx[i,:,:] = -(- 1.*f[NX-1,:,:] + 9.*f[NX-2,:,:] -45.*f[NX-3,:,:] \
                                               + 1.*f[NX-7,:,:] - 9.*f[NX-6,:,:] +45.*f[NX-5,:,:])/60.*self.scale[0]

        return self.dfdx

    def calc_dfdy(self, f):
    
        NY = self.NY

        for j in range(4):
            if j==0: self.dfdy[:,j,:] = (-11.*f[:,0,:] +18.*f[:,1,:] - 9.*f[:,2,:] + 2.*f[:,3,:])/ 6.*self.scale[1]
            if j==1: self.dfdy[:,j,:] = (- 2.*f[:,0,:] - 3.*f[:,1,:] + 6.*f[:,2,:] - 1.*f[:,3,:])/ 6.*self.scale[1]
            if j==2: self.dfdy[:,j,:] = (+ 1.*f[:,0,:] - 8.*f[:,1,:] - 1.*f[:,4,:] + 8.*f[:,3,:])/12.*self.scale[1]
            if j==3: self.dfdy[:,j,:] = (- 1.*f[:,0,:] + 9.*f[:,1,:] -45.*f[:,2,:] \
                                         + 1.*f[:,6,:] - 9.*f[:,5,:] +45.*f[:,4,:])/60.*self.scale[1]

        for j in range(4, NY-4):
            self.dfdy[:,j,:] = (   self.de *( f[:,j+4,:] - f[:,j-4,:] ) \
                                 + self.ce *( f[:,j+3,:] - f[:,j-3,:] ) \
                                 + self.be *( f[:,j+2,:] - f[:,j-2,:] ) \
                                 + self.ae *( f[:,j+1,:] - f[:,j-1,:] ) ) * self.scale[1]

            #self.dfdy[:,j,:] = 0.5*(f[:,j+1,:] - f[:,j-1,:])*self.scale[1]

        for j in range(NY-4, NY):
            if j==NY-1-0: self.dfdy[:,j,:] = -(-11.*f[:,NY-1,:] +18.*f[:,NY-2,:] - 9.*f[:,NY-3,:] + 2.*f[:,NY-4,:])/ 6.*self.scale[1]
            if j==NY-1-1: self.dfdy[:,j,:] = -(- 2.*f[:,NY-1,:] - 3.*f[:,NY-2,:] + 6.*f[:,NY-3,:] - 1.*f[:,NY-4,:])/ 6.*self.scale[1]
            if j==NY-1-2: self.dfdy[:,j,:] = -(+ 1.*f[:,NY-1,:] - 8.*f[:,NY-2,:] - 1.*f[:,NY-5,:] + 8.*f[:,NY-4,:])/12.*self.scale[1]
            if j==NY-1-3: self.dfdy[:,j,:] = -(- 1.*f[:,NY-1,:] + 9.*f[:,NY-2,:] -45.*f[:,NY-3,:] \
                                               + 1.*f[:,NY-7,:] - 9.*f[:,NY-6,:] +45.*f[:,NY-5,:])/60.*self.scale[1]

        return self.dfdy

#    //---------------------------------------
#    struct dfdy{
#      static_assert(NY==1 || NY>8, "CentDerv8 requires 9+ points");
#      StridedFieldVarType f; 
#      Field1DVarType df; 
#      double scale; 
#      dfdy(StridedFieldVarType f_, Field1DVarType df_, double scale_): f(f_), df(df_), scale(scale_) {}; 
#      KOKKOS_INLINE_FUNCTION
#      void operator()(const size_type n) const {
#        int k=n/NX/NY;
#        int j=(n-k*NX*NY)/NX;
#        int i=(n-k*NX*NY)%NX;
#        int l=k*NX*NY+j*NX+i;
#        j+=4; 
#        df(l) = ( de *( f(i,j+4,k)-f(i,j-4,k) )
#                + ce *( f(i,j+3,k)-f(i,j-3,k) )
#                + be *( f(i,j+2,k)-f(i,j-2,k) )
#                + ae *( f(i,j+1,k)-f(i,j-1,k) ) )*scale;
#      }
#    };

#    //---------------------------------------
#    struct dfdz{
#      static_assert(NZ==1 || NZ>8, "CentDerv8 requires 9+ points");
#      StridedFieldVarType f; 
#      Field1DVarType df; 
#      double scale; 
#      dfdz(StridedFieldVarType f_, Field1DVarType df_, double scale_): f(f_), df(df_), scale(scale_) {}; 
#      KOKKOS_INLINE_FUNCTION
#      void operator()(const size_type n) const {
#        int k=n/NX/NY;
#        int j=(n-k*NX*NY)/NX;
#        int i=(n-k*NX*NY)%NX;
#        int l=(k)*NX*NY+(j)*NX+(i);
#        k+=4; 
#        df(l) = ( de *( f(i,j,k+4)-f(i,j,k-4) )
#                + ce *( f(i,j,k+3)-f(i,j,k-3) )
#                + be *( f(i,j,k+2)-f(i,j,k-2) )
#                + ae *( f(i,j,k+1)-f(i,j,k-1) ) )*scale;
#      }
#    };

#    //---------------------------------------
#    struct dfdx_left_boundary{
#      StridedFieldVarType f;
#      Field1DVarType df;
#      double scale;
#      dfdx_left_boundary(StridedFieldVarType f_, Field1DVarType df_, double scale_): f(f_), df(df_), scale(scale_) {};
#      KOKKOS_INLINE_FUNCTION
#      void operator()(const size_type n) const {
#        int k=n/NX/NY;
#        int j=(n-k*NX*NY)/NX;
#        int i=(n-k*NX*NY)%NX;
#        int l=k*NX*NY+j*NX+i;
#        if(i==0) df(l) = (-11.*f(3+1,j,k) +18.*f(3+2,j,k) - 9.*f(3+3,j,k) + 2.*f(3+4,j,k))/ 6.*scale;
#        if(i==1) df(l) = (- 2.*f(3+1,j,k) - 3.*f(3+2,j,k) + 6.*f(3+3,j,k) - 1.*f(3+4,j,k))/ 6.*scale;
#        if(i==2) df(l) = (+ 1.*f(3+1,j,k) - 8.*f(3+2,j,k)
#                          - 1.*f(3+5,j,k) + 8.*f(3+4,j,k)                )/12.*scale;
#        if(i==3) df(l) = (- 1.*f(3+1,j,k) + 9.*f(3+2,j,k) -45.*f(3+3,j,k)
#                          + 1.*f(3+7,j,k) - 9.*f(3+6,j,k) +45.*f(3+5,j,k))/60.*scale;
#      }
#    };

#    //---------------------------------------
#    struct dfdx_right_boundary{
#      StridedFieldVarType f;
#      Field1DVarType df;
#      double scale;
#      dfdx_right_boundary(StridedFieldVarType f_, Field1DVarType df_, double scale_): f(f_), df(df_), scale(scale_) {};
#      KOKKOS_INLINE_FUNCTION
#      void operator()(const size_type n) const {
#        int k=n/NX/NY;
#        int j=(n-k*NX*NY)/NX;
#        int i=(n-k*NX*NY)%NX;
#        int l=k*NX*NY+j*NX+i;
#        if(i==NX-1-0) df(l) =-(-11.*f(NX+4-1,j,k) +18.*f(NX+4-2,j,k) - 9.*f(NX+4-3,j,k) + 2.*f(NX+4-4,j,k))/ 6.*scale;
#        if(i==NX-1-1) df(l) =-(- 2.*f(NX+4-1,j,k) - 3.*f(NX+4-2,j,k) + 6.*f(NX+4-3,j,k) - 1.*f(NX+4-4,j,k))/ 6.*scale;
#        if(i==NX-1-2) df(l) =-(+ 1.*f(NX+4-1,j,k) - 8.*f(NX+4-2,j,k)
#                               - 1.*f(NX+4-5,j,k) + 8.*f(NX+4-4,j,k)                   )/12.*scale;
#        if(i==NX-1-3) df(l) =-(- 1.*f(NX+4-1,j,k) + 9.*f(NX+4-2,j,k) -45.*f(NX+4-3,j,k)
#                               + 1.*f(NX+4-7,j,k) - 9.*f(NX+4-6,j,k) +45.*f(NX+4-5,j,k))/60.*scale;
#      }
#    };

#    //---------------------------------------
#    struct dfdy_left_boundary{
#      StridedFieldVarType f;
#      Field1DVarType df;
#      double scale;
#      dfdy_left_boundary(StridedFieldVarType f_, Field1DVarType df_, double scale_): f(f_), df(df_), scale(scale_) {};
#      KOKKOS_INLINE_FUNCTION
#      void operator()(const size_type n) const {
#        int k=n/NX/NY;
#        int j=(n-k*NX*NY)/NX;
#        int i=(n-k*NX*NY)%NX;
#        int l=k*NX*NY+j*NX+i;
#        if(j==0) df(l) = (-11.*f(i,3+1,k) +18.*f(i,3+2,k) - 9.*f(i,3+3,k) + 2.*f(i,3+4,k))/ 6.*scale;
#        if(j==1) df(l) = (- 2.*f(i,3+1,k) - 3.*f(i,3+2,k) + 6.*f(i,3+3,k) - 1.*f(i,3+4,k))/ 6.*scale;
#        if(j==2) df(l) = (+ 1.*f(i,3+1,k) - 8.*f(i,3+2,k)
#                          - 1.*f(i,3+5,k) + 8.*f(i,3+4,k)                )/12.*scale;
#        if(j==3) df(l) = (- 1.*f(i,3+1,k) + 9.*f(i,3+2,k) -45.*f(i,3+3,k)
#                          + 1.*f(i,3+7,k) - 9.*f(i,3+6,k) +45.*f(i,3+5,k))/60.*scale;
#      }
#    };
#    //---------------------------------------
#    struct dfdy_right_boundary{
#      StridedFieldVarType f;
#      Field1DVarType df;
#      double scale;
#      dfdy_right_boundary(StridedFieldVarType f_, Field1DVarType df_, double scale_): f(f_), df(df_), scale(scale_) {};
#      KOKKOS_INLINE_FUNCTION
#      void operator()(const size_type n) const {
#        int k=n/NX/NY;
#        int j=(n-k*NX*NY)/NX;
#        int i=(n-k*NX*NY)%NX;
#        int l=k*NX*NY+j*NX+i;
#        if(j==NY-1-0) df(l) =-(-11.*f(i,NY+4-1,k) +18.*f(i,NY+4-2,k) - 9.*f(i,NY+4-3,k) + 2.*f(i,NY+4-4,k))/ 6.*scale;
#        if(j==NY-1-1) df(l) =-(- 2.*f(i,NY+4-1,k) - 3.*f(i,NY+4-2,k) + 6.*f(i,NY+4-3,k) - 1.*f(i,NY+4-4,k))/ 6.*scale;
#        if(j==NY-1-2) df(l) =-(+ 1.*f(i,NY+4-1,k) - 8.*f(i,NY+4-2,k)
#                               - 1.*f(i,NY+4-5,k) + 8.*f(i,NY+4-4,k)                   )/12.*scale;
#        if(j==NY-1-3) df(l) =-(- 1.*f(i,NY+4-1,k) + 9.*f(i,NY+4-2,k) -45.*f(i,NY+4-3,k)
#                               + 1.*f(i,NY+4-7,k) - 9.*f(i,NY+4-6,k) +45.*f(i,NY+4-5,k))/60.*scale;
#      }
#    };

#    //---------------------------------------
#    struct dfdz_left_boundary{
#      StridedFieldVarType f;
#      Field1DVarType df;
#      double scale;
#      dfdz_left_boundary(StridedFieldVarType f_, Field1DVarType df_, double scale_): f(f_), df(df_), scale(scale_) {};
#      KOKKOS_INLINE_FUNCTION
#      void operator()(const size_type n) const {
#        int k=n/NX/NY;
#        int j=(n-k*NX*NY)/NX;
#        int i=(n-k*NX*NY)%NX;
#        int l=k*NX*NY+j*NX+i;
#        if(k==0) df(l) = (-11.*f(i,j,3+1) +18.*f(i,j,3+2) - 9.*f(i,j,3+3) + 2.*f(i,j,3+4))/ 6.*scale;
#        if(k==1) df(l) = (- 2.*f(i,j,3+1) - 3.*f(i,j,3+2) + 6.*f(i,j,3+3) - 1.*f(i,j,3+4))/ 6.*scale;
#        if(k==2) df(l) = (+ 1.*f(i,j,3+1) - 8.*f(i,j,3+2)
#                          - 1.*f(i,j,3+5) + 8.*f(i,j,3+4)                )/12.*scale;
#        if(k==3) df(l) = (- 1.*f(i,j,3+1) + 9.*f(i,j,3+2) -45.*f(i,j,3+3)
#                          + 1.*f(i,j,3+7) - 9.*f(i,j,3+6) +45.*f(i,j,3+5))/60.*scale;
#      }
#    };
#    //---------------------------------------
#    struct dfdz_right_boundary{
#      StridedFieldVarType f;
#      Field1DVarType df;
#      double scale;
#      dfdz_right_boundary(StridedFieldVarType f_, Field1DVarType df_, double scale_): f(f_), df(df_), scale(scale_) {};
#      KOKKOS_INLINE_FUNCTION
#      void operator()(const size_type n) const {
#        int k=n/NX/NY;
#        int j=(n-k*NX*NY)/NX;
#        int i=(n-k*NX*NY)%NX;
#        int l=k*NX*NY+j*NX+i;
#        if(k==NZ-1-0) df(l) =-(-11.*f(i,j,NZ+4-1) +18.*f(i,j,NZ+4-2) - 9.*f(i,j,NZ+4-3) + 2.*f(i,j,NZ+4-4))/ 6.*scale;
#        if(k==NZ-1-1) df(l) =-(- 2.*f(i,j,NZ+4-1) - 3.*f(i,j,NZ+4-2) + 6.*f(i,j,NZ+4-3) - 1.*f(i,j,NZ+4-4))/ 6.*scale;
#        if(k==NZ-1-2) df(l) =-(+ 1.*f(i,j,NZ+4-1) - 8.*f(i,j,NZ+4-2)
#                               - 1.*f(i,j,NZ+4-5) + 8.*f(i,j,NZ+4-4)                   )/12.*scale;
#        if(k==NZ-1-3) df(l) =-(- 1.*f(i,j,NZ+4-1) + 9.*f(i,j,NZ+4-2) -45.*f(i,j,NZ+4-3)
#                               + 1.*f(i,j,NZ+4-7) - 9.*f(i,j,NZ+4-6) +45.*f(i,j,NZ+4-5))/60.*scale;
#      }
#    };
#};
