import numpy as np
import sys

from amuse.units import units, constants
from amuse.datamodel import Particles, Particle, new_regular_grid

from shared_src.extra_funcs import dynamical_mass, Rhills
from shared_src import OL18


class PebbleGasAccretion:
    """Class to handle planetary pebble and gas accretion."""
    def __init__(self, ndisk_cells):
        self.planets = Particles()
        self.planets.add_calculated_attribute(
            'dynamical_mass', dynamical_mass
            )

        self.star = Particle(mass=1|units.MSun)  #DEBUG - Why is it MSun
        self.model_time = 0. | units.Myr

        # setup disk
        self.disk = new_regular_grid(([int(ndisk_cells)]),[1])
        self.disk.surface_gas = 0 | units.g/units.cm**2
        self.disk.surface_solid = 0 | units.g/units.cm**2
        self.disk.scale_height = 0.03 * self.disk.position

        # DEBUG - Why is this different to numbers in params.py? Planets versus Star?
        self.disk.alpha_acc = 1e-3
        self.disk.alpha = 1e-4
        self.disk.st = 0.01  
        self.disk.vd = 0 | units.cm/units.s
        
        # planet atmosphere
        self.kappa = 0.005 | units.m**2/units.kg

        self.BL = 1 #feeding zone size =6 for planetesimal accretion
        self.eta = 0.1
        

    def feedzone(self):
        Sigmadmean = np.zeros(len(self.planets)) | units.g/units.cm**2
        Mfeed = np.zeros(len(self.planets)) | units.g
        imax = np.zeros(len(self.planets))
        imin = np.zeros(len(self.planets))
        ip = np.zeros(len(self.planets))
        RHs = np.zeros(len(self.planets)) | units.au
        Rdisk = self.disk.position

        Sigmad = self.disk.surface_solid
        fliss = self.BL

        for i in range(len(self.planets)):
            ap = self.planets[i].semimajor_axis
            RH = Rhills(self.planets[i].core_mass,self.star.mass,ap)
            RHs[i] = RH
            Rmin=ap-fliss*RH
            Rmax=ap+fliss*RH

            Mfeedi = 0. | units.g

            if (Rmin<Rdisk[-1]) and (Rmin>Rdisk[0]):
                imini = np.nonzero(Rdisk<=Rmin)[0][-1]
            elif (Rmin<=Rdisk[0]):
                imini = 0
            else:
                imini = -2

            if (Rmax<Rdisk[-1]) and (Rmax>Rdisk[0]):
                imaxi = np.nonzero(Rdisk<=Rmax)[0][-1]
            elif (Rmax<=Rdisk[0]):
                imaxi = 0
            else:
                imaxi = -2

            ipi   = np.nonzero(Rdisk<=ap)[0][-1]
            
            # improved from mordasini2015
            if imaxi == imini:
                imaxi = imini+1

            for j in range(imini, imaxi):
                Mfeedi += np.pi* (Rdisk[j+1]**2-Rdisk[j]**2)*(Sigmad[j+1]+Sigmad[j])/2

            Sigmadmeani=Mfeedi/(np.pi*(Rdisk[imaxi]**2-Rdisk[imini]**2))
            
            Sigmadmean[i] = Sigmadmeani
            Mfeed[i] = Mfeedi
            imax[i] = imaxi
            imin[i] = imini
            ip[i] = ipi
        return Sigmadmean, Mfeed, imax, imin, ip, RHs

    def Mdotcore(self):
        mdotc = np.zeros(len(self.planets)) | units.MEarth/units.kyr
        Sigmadmean, Mfeed, imax, imin, ip, RHs = self.feedzone()
        Mstar = self.star.mass
        misos = np.zeros(len(self.planets)) | units.MEarth
        Hdisk = self.disk.scale_height
        Rdisk = self.disk.position
        sigmad = self.disk.surface_solid
        sigmag = self.disk.surface_gas

        for i in range(len(self.planets)):
            ap = self.planets[i].semimajor_axis
            mc = self.planets[i].core_mass
            
            #calculate slope
            xi = - np.log(Hdisk[int(ip[i])+1]/Hdisk[int(ip[i])])/np.log(Rdisk[int(ip[i])+1]/Rdisk[int(ip[i])])*2+3
            beta = - np.log(sigmag[int(ip[i])+1]/sigmag[int(ip[i])])/np.log(Rdisk[int(ip[i])+1]/Rdisk[int(ip[i])])

            H = (Hdisk[int(ip[i])]-Hdisk[int(ip[i])+1])*(Rdisk[int(ip[i])]-ap)/(Rdisk[int(ip[i])]-Rdisk[int(ip[i])+1])+Hdisk[int(ip[i])]
            miso = (25|units.MEarth)*(0.34*(-3/np.log10(self.disk.alpha[int(ip[i])]))**4+0.66)*(1-(-1.5-0.5*xi-beta+2.5)/6)*(H/ap/0.05)**3 * (Mstar.value_in(units.MSun))

            misos[i] = miso
            sigmadp = (sigmad[int(ip[i])]-sigmad[int(ip[i])+1])*(Rdisk[int(ip[i])]-ap)/(Rdisk[int(ip[i])]-Rdisk[int(ip[i])+1])+sigmad[int(ip[i])]
            if (mc>=miso) or (self.planets[i].isohist==True):
                mdotc[i] = 0 | units.g/units.s
                self.planets[i].isohist = True

            else:
                if (sigmadp<1e-20|units.g/units.cm**2):
                    mdotc[i] = 0 | units.g/units.s
                else:
                    eta = -0.5* (H/ap)**2 * (-1.5-0.5*xi-beta)
                    hgas = H/ap
                    Vk = np.sqrt(constants.G*Mstar/ap)
                    epsilon = OL18.epsilon(ep=0, tau=self.disk.st[int(ip[i])], qp=mc/Mstar, eta = eta, hgas=hgas, alphaz=self.disk.alpha[int(ip[i])], Rp=(1|units.REarth)/ap)
                    epsilon = np.minimum(epsilon,1)
                    if self.disk.vd[int(ip[i])].value_in(units.cm/units.s) == 0.:
                        self.disk.vd[int(ip[i])] = -2* self.disk.st[int(ip[i])]* eta * Vk
                        self.disk.vd[int(ip[i])] -= 3/2*self.disk.alpha_acc[int(ip[i])] * hgas**2* Vk  # motion is relative to the gas accretion
                    mdotc[i] = epsilon*2*np.pi* ap * abs(self.disk.vd[int(ip[i])]) * sigmadp  # pebble accretion Johansen et al. (2019)

        return mdotc, Sigmadmean, Mfeed, imax, imin, ip, misos

    def Mdotgas(self, misos, ip):
        if len(misos) != len(self.planets):
            print('dimension is not consistent!')
            sys.exit(0)
        mdotg = np.zeros(len(self.planets)) | units.g/units.s
        mass_p = self.planets.dynamical_mass
        ap = self.planets.semimajor_axis
        kappa = self.kappa
        Sigmag = self.disk.surface_gas
        Mstar = self.star.mass
        Hdisk = self.disk.scale_height
        Rdisk = self.disk.position
        for i in range(len(self.planets)):
            M_crit = misos[i]
            alpha = self.disk.alpha[int(ip[i])]
            alpha_acc = self.disk.alpha_acc[int(ip[i])]
            if mass_p[i] < M_crit:  
                mdotg[i] = 0 | units.g/units.s
            else:
                mdotg[i] = (10**-5|units.MEarth/units.yr)*(mass_p[i].value_in(units.MEarth)/10)**4*(kappa/(0.1|units.m**2/units.kg))**(-1) # Johansen2019; Ikoma2000

                omega = np.sqrt(constants.G*Mstar/ap[i]**3)
                H = (Hdisk[int(ip[i])]-Hdisk[int(ip[i])+1])*(Rdisk[int(ip[i])]-ap[i])/(Rdisk[int(ip[i])]-Rdisk[int(ip[i])+1])+Hdisk[int(ip[i])]

                h=H/ap[i]
                q = mass_p[i]/Mstar
                K = (q)**2*(h)**-5/alpha

                mdotbondi = 0.29/np.pi/3*(h)**(-4)*(q)**(4/3)/(1+0.04*K)/alpha_acc # Johansen2019; Tanigawa & Tanaka2016
                ff = min(mdotbondi,1) # min(mgdot, mgdot_bondi)
                
                nudisk = alpha_acc*(h*ap[i])**2*omega
                Mdotdisk = 3*np.pi*nudisk*Sigmag[int(ip[i])]
                mdotg[i] = min(mdotg[i],ff*Mdotdisk)

        return mdotg

    def set_time_scale(self, Mdotcore, Mdotenvelop, model_time_i, end_time):
        dt_core = 0.01*min(self.planets.core_mass/Mdotcore)
        dt_env = 0.01*min(self.planets.envelope_mass/Mdotenvelop)
        dt = abs(min(dt_core, dt_env, model_time_i- end_time))
        return dt
    
    def evolve_model(self, end_time):
        model_time_i = self.model_time
        Rdisk = self.disk.position
        Sigmad = self.disk.surface_solid

        if (min(self.planets.semimajor_axis)>Rdisk[0]) and (max(self.planets.semimajor_axis)<Rdisk[-1]):
            while model_time_i < end_time: # improved from pps.py
            
                Mdotcore, Sigmadmean, Mfeed, imax, imin, ip, misos = self.Mdotcore()
                Mdotenvelop = self.Mdotgas(misos, ip)
                dt = self.set_time_scale(Mdotcore, Mdotenvelop, model_time_i, end_time)
                model_time_i += dt

                for i in range(len(self.planets)):
                    self.planets[i].core_mass += Mdotcore[i] * dt
                    if self.planets[i].isohist==False:
                        self.planets[i].core_mass = min(self.planets[i].core_mass, misos[i])
                    self.planets[i].envelope_mass += Mdotenvelop[i] * dt # gas accretion
        else:
            pass

        self.model_time = end_time # improved from pps.py
        self.disk.surface_solid = Sigmad

    def diskd_mass(self):
        Mddisk = 0|units.g
        Rdisk = self.disk.position
        Sigmad = self.disk.surface_solid
        for i in range(len(Rdisk)-1):
            Mddisk+=np.pi*(Rdisk[i+1]**2-Rdisk[i]**2)*0.5*(Sigmad[i+1]+Sigmad[i])
        return Mddisk

    def diskg_mass(self):
        Mgdisk = 0|units.g
        Rdisk = self.disk.position
        Sigmag = self.disk.surface_gas
        for i in range(len(Rdisk)-1):
            Mgdisk += np.pi*(Rdisk[i+1]**2-Rdisk[i]**2)*0.5*(Sigmag[i+1]+Sigmag[i])
        return Mgdisk
