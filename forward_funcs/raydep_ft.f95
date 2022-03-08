subroutine raydep_ft(v, u, zz, zx, p, om, model, n_layers)
  !
  ! Translation of Wayne Crawford's MATLAB code for forward calculating 
  ! normalized compliance from synthetic Earth structure to Fortran95.
  !
  ! This implementation is much faster than the original MATLAB code
  !
  ! Original source can be found at (http://www.ipgp.fr/~crawford/Homepage/Software.html).
  !
  ! Stephen Mosher, June 2020 

  ! Define all variables
  implicit none

  ! Variable declarations (in/out)
  integer,           intent(in)  :: n_layers
  double precision,  intent(in)  :: p, om
  double precision,  intent(in)  :: model(n_layers, 4)
  double precision, intent(out)  :: v(n_layers), u(n_layers), zz(n_layers), zx(n_layers)


  ! Variable declarations (intermediate)
  double precision               :: d(n_layers)
  double precision               :: rho(n_layers)
  double precision               :: Vp(n_layers)
  double precision               :: Vs(n_layers)
  double precision               :: mu(n_layers)
  double precision               :: y(5), ym(n_layers, 5) 
  double precision               :: x(n_layers,4)
  double precision               :: RoW, SoW, r1, r2, ynorm
  double precision               :: ha, hb, ca, cb, sa, sb, hbs, has, sum, pbsq
  double precision               :: b1, g1, g2, g3, e1, e2, e3, e4, e6, e8
  integer                        :: ist, i, ls

  ! Some initializations
  ca = 0
  cb = 0
  sa = 0
  sb = 0

  ! Extract model parameters
  d = model(:,1)             ! layer thicknesses
  rho = model(:,2)           ! layer density
  Vp = model(:,3)            ! P-wave velocity
  Vs = model(:,4)            ! S-wave velocity 

  ! Convert model parameters to SI units
  ! d                        ! layer thicknesses already in m
  rho = rho * 1000           ! convert from [g/cm^3] to [kg/m^3]
  Vp = Vp * 1000             ! convert from [km/s] to [m/s]
  Vs = Vs * 1000             ! convert from [km/s] to [m/s]

  ! Some intermediate parameters
  mu = rho * Vs**2
  ist = n_layers
  r2 = 2 * mu(ist) * p

  ! R and S are the "Wavenumbers" of compress and shear waves in botlayer
  ! RoW and SoW are divided by ang freq
  RoW = sqrt(p**2 - 1/Vp(ist)**2)
  SoW = sqrt(p**2 - 1/Vs(ist)**2)
  i = ist

  y(1) = (RoW * SoW - p**2)/rho(i)
  y(2) = r2 * y(1) + p
  y(3) = RoW
  y(4) = -SoW
  y(5) = rho(i) - r2 * (p + y(2))

  ym(i, 1) = y(1)
  ym(i, 2) = y(2)
  ym(i, 3) = y(3)
  ym(i, 4) = y(4)
  ym(i, 5) = y(5)
  
  !*****PROPAGATE UP LAYERS*********
  do while (i .gt. 1)
    i = i - 1
    
    ha = p**2 - 1 / Vp(i)**2
    call argdtray(om * d(i), ha, ca, sa)
    
    hb = p**2 - 1/Vs(i)**2
    call argdtray(om * d(i), hb, cb, sb)
  
    hbs = hb * sb
    has = ha * sa
    r1 = 1 / rho(i)
    r2 = 2 * mu(i) * p
    b1 = r2 * y(1) - y(2)
    g3 = ( y(5) + r2 * (y(2) - b1) ) * r1
    g1 = b1 + p * g3
    g2 = rho(i) * y(1) - p * (g1 + b1)
    e1 = cb * g2 - hbs * y(3)
    e2 = -sb * g2 + cb * y(3)
    e3 = cb * y(4) + hbs * g3
    e4 = sb * y(4) + cb * g3
    y(3) = ca * e2 - has * e4
    y(4) = sa * e1 + ca * e3
    g3 = ca * e4 - sa * e2
    b1 = g1 - p * g3
    y(1) = (ca * e1 + has * e3 + p * (g1 + b1)) * r1
    y(2) = r2 * y(1) - b1
    y(5) = rho(i) * g3 - r2 * (y(2) - b1)
    ym(i,:) = y
  end do
  
  ynorm = 1 / y(3)
  y(1) = 0
  y(2) = -ynorm
  y(3) = 0
  y(4) = 0
  
  !*****PROPAGATE BACK DOWN LAYERS*********
  do while (i .le. ist)
    x(i,1) = -ym(i,2) * y(1) - ym(i,3) * y(2) + ym(i,1) * y(4)
    x(i,2) = -ym(i,4) * y(1) + ym(i,2) * y(2) - ym(i,1) * y(3)
    x(i,3) = -ym(i,5) * y(2) - ym(i,2) * y(3) - ym(i,4) * y(4)
    x(i,4) =  ym(i,5) * y(1) - ym(i,3) * y(3) + ym(i,2) * y(4)
    ls = i
    
    if (i .ge. 2) then
      sum = abs( x(i,1) + i * x(i,2))
      pbsq = 1 / Vs(i)**2
      if (sum .lt. 1e-4) then
        exit ! ########################### supposed to be MATLAB break
      end if
    end if

    ha = p**2 - 1/Vp(i)**2
    call argdtray(om * d(i), ha, ca, sa)
    
    hb = p**2 - 1/Vs(i)**2
    call argdtray(om * d(i), hb, cb, sb)
    
    hbs = hb * sb
    has = ha * sa
    r2 = 2 * p * mu(i)
    e2 = r2 * y(2) - y(3)
    e3 = rho(i) * y(2) - p * e2
    e4 = r2 * y(1) - y(4)
    e1 = rho(i) * y(1) - p * e4
    e6 = ca * e2 - sa * e1
    e8 = cb * e4 - sb * e3
    y(1) = (ca * e1 - has * e2+p * e8) / rho(i)
    y(2) = (cb * e3 - hbs * e4+p * e6) / rho(i)
    y(3) = r2 * y(2) - e6
    y(4) = r2 * y(1) - e8
    i = i + 1
  end do

  if (x(1,3) .eq. 0) then
    !error('vertical surface stress = 0 in DETRAY') ! Fix this
  end if

  ist = ls
  v =  x(:,1)
  u =  x(:,2)
  zz = x(:,3)
  zx = x(:,4)

end subroutine raydep_ft

subroutine argdtray(om, h, c, s)
  !
  ! Translation of Wayne's code to Fortran95
  ! 

  ! Define all variables
  implicit none

  ! Variable declarations (in/out)
  double precision,      intent(in)  :: om, h
  double precision,     intent(out)  :: c, s

  ! Intermediate variables
  double precision                   :: hh, th, d

  ! Function proper
  hh = sqrt(abs(h)) 
  th = om * hh    ! number of waves (or e-foldings) in layer in radians
  if (th .ge. 1.5E-14) then
    if (h .le. 0) then   ! propagating wave
      c =  cos(th)
      s = -sin(th) / hh
    else if (h .gt. 0) then
      d = exp(th)
      c =  0.5 * (d + 1/d)
      s = -0.5 * (d - 1/d)/hh
    end if
  else if (th .lt. 1.5E-14) then
      c = 1
      s = -om
  end if
end subroutine argdtray