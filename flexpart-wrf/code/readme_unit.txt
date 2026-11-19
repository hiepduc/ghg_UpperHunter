(1) The CONC field is built from:

$$ \frac{xmass}{\rho}\times weight $$

weight is a time-integration weight, not a source-emission normalization. The actual release mass is stored in xmass, then divided among the 500 particles

(2) For example
500
1.0
Singleton

the 1.0 is directly read as:

total released mass = xmass = 1.0 for AIRTRACER, distributed among 500 particles.

And we already established from releaseparticles_reg.f90 that each particle gets approximately:

$$ xmass_{\rm particle}=\frac{1.0}{500}=0.002 $$

before the time-correction factor.

We now know:

tot_mu_scalar  ! total mass per source and species (backward)

and FLEXPART computes

$$ CONC = grid \times \frac{factor3d}{tot\_mu\_scalar}. $$

(3)
The release has:

xmass = 1.0

so the backward footprint is normalized by the 1.0 kg total release mass.


(4) and the source-code comment explicitly says that in a backward simulation:

release = receptor, sampling = source.

the CONC field is the backward source–receptor response function, with units s m³ kg⁻¹.

(5) for the backward run (ldirect = -1), FLEXPART uses:

factor3d(ix,jy,kz) = real(abs(loutaver))/outnum

and then:

grid2 = grid * factor3d / tot_mu_scalar

with:

tot_mu_scalar = total mass per source and species (backward)

The release has xmass = 1.0, so the normalization is indeed to a 1 kg tracer release.

The resulting CONC units:

s m3 kg-1

are therefore consistent with the backward source–receptor sensitivity.

(5) FLEXPART footprint is a 3-D field, and the mine emissions are distributed over an area. We need to establish exactly how the source emission is represented when converting the 3-D FLEXPART sensitivity into a receptor concentration.

In particular, we need to resolve:

horizontal grid-cell area,
vertical layer thickness,
whether the mine inventory is total kg/s or an areal flux kg m⁻² s⁻¹,
how FLEXPART's CONC relates to a distributed source.

Next diagnostic: inspect the source/output equation

(6) For your backward run, FLEXPART writes each grid-cell concentration as

$$ CONC_{t,x,y,z} = grid_{t,x,y,z} \frac{|loutaver|/outnum}{tot\_mu}. $$

And from conccalc_reg.f90,

grid
t,x,y,z ∼ i ∑ ρ i xmass i weight i w i

(7) THIS IS IMPORTANT

For the backward FLEXPART footprint,

{CONC} = sm^3kg^{-1}

can be understood as a residence-time sensitivity per unit source mass.

The three units mean:

s = accumulated particle residence time
m³ = the air volume associated with the grid cell
kg⁻¹ = normalized by the mass released in the backward simulation

So, conceptually:

CONC∼
released tracer mass
particle residence time in a grid-cell volu

Thus a high FLEXPART CONC means:

Air arriving at the receptor has a strong sensitivity to emissions from that location because the backward particles spend more time there.

This is exactly why the backward FLEXPART field can be used as a source–receptor sensitivity.

(8)  The FLEXPART output has two vertical layers:

$$ \boxed{\text{Layer 1: surface–100 m}} $$ $$ \boxed{\text{Layer 2: 100–20,000 m}} $$

because:

ZTOP = [100, 20000] m

and the mine emissions are surface emissions.

So for the mine source, the physically relevant FLEXPART footprint is the 0–100 m layer, i.e. bottom_top = 0 in Python indexing.

(8) So the NetCDF CONC is directly derived from the accumulated particle residence field grid.

For the backward run:

factor3d = abs(loutaver) / outnum

Therefore:

$$ \boxed{ CONC = \frac{grid}{M_{\rm release}} \frac{|loutaver|}{N_{\rm output}} } $$

with units:

sm^3kg^{-1}

for the ind_rel=0, ind_samp=-1 configuration.

