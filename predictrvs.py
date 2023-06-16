import numpy as np 
import pandas as pd 
import radvel 

class Observations(object):
    def __init__(self, times, rvs, rverrs, telescope):
        assert times.shape == rvs.shape == rverrs.shape, "Time, RV and RV Error arrays must have the same shape"
        self.times = times
        self.rvs = rvs
        self.rverrs = rverss
        self.telescope = telescope
        self.nobs = times.shape[0]

    def __repr__(self):
        return f'({self.telescope}: {self.nobs} Observations)'



class HostStar(object):
    """
    This class represents the host star in each planetary system
    """

    def __init__(self, mass, masserr, radius, radiuserr, teff, tefferr):
        """
        Initialize HostStar object

        Args:
            mass (float): Mstar in units of Msun
            masserr (float): Uncertainty in Mstar in units of Msun
            radius (float): Rstar in units of rsun
            radiuserr (float): Uncertainty in Rstar in units of Rsun
            teff (int): Effective temperature in Kelvin
            tefferr (int): Uncertainty in Teff in Kelvin
        """

        self.mass = mass
        self.masserr = masserr
        self.radius = radius
        self.radiuserr = radiuserr
        self.teff = teff
        self.tefferr = tefferr

    def __repr__(self):
        """
        Creates representation of host star with mass, and radius for readability

        Returns:
            str : Host star mass and radius
        """

        return f'(Star: {self.mass} Msun, {self.radius} Rsun)'


class Planet(object):
    """
    This class represents a planet in a given system.

    """

    def __init__(self, letter, P, t0, K, ecc=0, omega=0, Perr=0, t0err=0, mass=0, masserr=0, radius=0, radiuserr=0):
        """
        Initialize Planet object

        Args:
            letter (str): Planet letter, typically but not always in order of semi-major axis in the system
            mass (float): Planet mass in units of Mearth
            masserr (float): Planet mass uncertainty in units of Mearth
            radius (float): Planet radius in units of Rearth
            radiuserr (float): Planet radius uncertainty in units of Rearth
            period (float): Planet orbital period in units of days (assuming linear emphemeris)
            perioderr (float): Planet orbital period uncertainty in units of days
            t0 (float): Planet transit epoch aka time of inferior conjunction in units of Julian Days (JD)
            t0err (float): Uncertainty in t0 in units of JD
            ecc (float): Planet eccentricity, between 0 and 1
            omega (float): Planet argument of periastron in units of radians
            K (float): Planet radial velocity semi-ampltiude in units of m/s

        """

        self.letter = letter
        self.mass = mass
        self.masserr = masserr
        self.radius = radius
        self.radiuserr = radiuserr
        self.period = P
        self.perioderr = Perr
        self.t0 = t0
        self.t0err = t0err
        self.ecc = ecc
        self.omega = omega
        self.K = K

    def __repr__(self):
        """
        Creates representation of planet with it's letter, mass, and radius for readability

        Returns:
            str : Planet letter, mass, and radius
        """
        return f'(Planet {self.letter}: {self.mass} Me, {self.radius} Re)'


@dataclass(init=True)  # init=True generates standardized __init__ method
class System:
    name: str  # Name of host star
    num_planets: int  # Number of planets in the system
    star: HostStar  # Host Star object
    planets: list  # List of Planet objects

    def __post_init__(self):
        """
        Check that the object is instantialized self-consistently

        Raises:
            AssertionError: If the system is not self-consistent, or has fewer than 2 planets

        """

        # For this science case, we only want to consider systems with 2+ planets
        assert self.num_planets >= 2, "Make sure you have at least 2 planets!"
        assert len(self.planets) == self.num_planets, "Make sure you've added all of the planets!"

    def make_setup_file(self, data_file, setup_file):
        """
        This function generates a setup file that can be read into and used by the Radvel package to perform an RV fit

        Args:
            data_file (str): Name of .csv file that contains the RV data
            setup_file (str): Name to save the setup file under

        """

        list_imports = ['import numpy as np', 'import radvel',
                        'import pandas as pd', 'import string', 'from matplotlib import rcParams']
        with open(setup_file, 'w') as file:
            # Import packages
            for imp in list_imports:
                file.write(imp+'\n')
            # Read in RV data, errors, time stamps, and telescope names into different arrays
            file.write(f"\ndata = pd.read_csv('{data_file}')\n\n")
            file.write("t = np.array(data.time)\nvel = np.array(data.mnvel)\nerrvel = np.array(data.errvel)\ntel = np.array(data.tel)\ntelgrps = data.groupby('tel').groups\ninstnames = telgrps.keys()\n\n")

            # Define system parameters including name, number of planets, and what basis we want to fit the data in
            file.write(
                f"starname = '{self.name}'\nnplanets = {self.num_planets}\n fitting_basis = 'per tc secosw sesinw k'\nbjd0 = 0.\n")

            # Generate a dictionary that maps the planet letter to a number as per Radvel formatting
            planet_letters = [pl.letter for pl in self.planets]
            planet_nums = [int(i) for i in np.arange(self.num_planets)+1]
            planet_letters = dict(zip(planet_nums, planet_letters))
            # List of telescopes used to collect data
            file.write(f"planet_letters = {planet_letters}\ntelescopes = np.unique(tel)\n\n")
            # Initialize fitting parameters
            file.write(
                f"params = radvel.Parameters({self.num_planets}, basis='per tc e w k', planet_letters=planet_letters)\n")
            # For each planet, initialize a period, time of inferior conjuction, eccentricity, argument of periastron, and RV semi-ampltiude using the values read in to the script
            for i, planet in enumerate(self.planets):
                i += 1
                file.write(f"params['per{i}'] = radvel.Parameter(value = {planet.period})\n")
                file.write(f"params['tc{i}'] = radvel.Parameter(value = {planet.t0})\n")
                # For simplicity, and motivated by Yee et al. 2021, set eccentricities to 0
                file.write(f"params['e{i}'] = radvel.Parameter(value = {planet.ecc}, vary=False)\n")
                file.write(f"params['w{i}'] = radvel.Parameter(value = {planet.omega})\n")
                file.write(f"params['k{i}'] = radvel.Parameter(value = {planet.K})\n\n")

            # Initialize parameters describing the global RV slope and curvature
            file.write("params['dvdt'] = radvel.Parameter(value=0.0)\n")
            file.write("params['curv'] = radvel.Parameter(value=0.0)\n\n")
            # For each telescope used, initialize an offset and a jitter instrumental term and set initial non-zero guesses
            file.write(f"for telescope in telescopes:\n")
            file.write("\tparams[f'gamma_{telescope}'] = radvel.Parameter(value=0.5, vary=True)\n")
            file.write("\tparams[f'jit_{telescope}'] = radvel.Parameter(value=3, vary=True)\n\n")
            # Transform the parameter basis to the fitting basis parameterisation (to simplify initialization)
            file.write("params = params.basis.to_any_basis(params, fitting_basis)\n")
            # Time which dvdt and curv are calculated relative to
            file.write("time_base = 2458989.783463\n")
            # Create RV model
            file.write("mod = radvel.RVModel(params, time_base=time_base)\n")

            # For each planet parameter, set whether it is allowed to vary in the fitting process
            for i, planet in enumerate(self.planets):
                i += 1
                file.write(f"mod.params['per{i}'].vary = False\n")
                file.write(f"mod.params['tc{i}'].vary = False\n")
                file.write(f"mod.params['secosw{i}'].vary = False\n")
                file.write(f"mod.params['sesinw{i}'].vary = False\n\n")

            # The same for the global RV parameters
            file.write("mod.params['dvdt'].vary = True\nmod.params['curv'].vary = False\n\n")
            # Set a prior to keep K > 0 - this can be inappropriate as it biases larger values of K
            file.write(f"priors = [radvel.prior.PositiveKPrior({self.num_planets})]\n")
            # Set priors on the planet parameters based on prior knowledge of uncertainties
            for i, planet in enumerate(self.planets):
                i += 1
                # Do not set a prior on period and t0 if there is no associated error as this breaks the fit
                if planet.perioderr != 0 and planet.t0err != 0:
                    file.write(
                        f"priors += [radvel.prior.Gaussian('per{i}', {planet.period}, {planet.perioderr})]\npriors += [radvel.prior.Gaussian('tc{i}', {planet.t0}, {planet.t0err})]\n")
            # Set a hard bound prior on instrumental parameters to speed up the fit
            file.write(f"for telescope in telescopes:\n")
            file.write("\tpriors += [radvel.prior.HardBounds(f'jit_{telescope}', -20.0, 20.0)]\n\n")
            # Add stellar parameters
            file.write(f"stellar = dict(mstar={self.star.mass}, mstar_err={self.star.masserr})")

def run_radvel(row):
    """
    Read in planetary system information for individual planets and stars
    Create variations of the systems with additional planets at various periods
    Fit the radial velocity data assuming various system parameters
    Compare the resulting masses for each of the known planets across the system variations

    Args:
        row (Pandas Series): row of a dataframe including all of the stellar host and planet parameters

    """
    # Create system object with host star and planets for the 'default' scenario
    star = HostStar(row['ms'], row['mserr'], row['rs'], row['rserr'], row['teff'], row['tefferr'])
    num_planets = row['npl']
    planets = []
    for i in range(1, num_planets+1):
        # Add planets
        pl = Planet(string.ascii_lowercase[i], row[f'p{i}'], row[f't0{i}'],
                    row[f'K{i}'], mass=row[f'mp{i}'], masserr=row[f'mperr{i}'], radius=row[f'rp{i}'], radiuserr=row[f'rperr{i}'], Perr=row[f'perr{i}'], t0err=row[f't0err{i}'], ecc=row[f'e{i}'], omega=row[f'w{i}'])
        planets.append(pl)
    sys = System(row['name'], num_planets, star, planets)

    # Create system object with host star and planets for the 'default' scenario
    star = HostStar(0.86, 0.12, 0.87, 0.1, 5151, 100)
    pl1 = Planet('b', 4.31, 2458686.5658, 3, 0, 0, 0.00002, 0.001, 8.1, 1.1, 3.01, 0.06)
    pl2 = Planet('c', 5.90, 2458683.4661, 3, 0, 0, 0.00008, 0.003, 8.8, 1.2, 2.51, 0.08)
    pl3 = Planet('d', 18.66, 2458688.9653, 3, 0, 0, 0.00005, 0.009, 5.3, 1.7, 3.51, 0.09)
    pl4 = Planet('e', 37.92, 2457000.7134, 3, 0, 0, 0.0001, 0.0089, 14.8, 2.3, 3.78, 0.16)
    pl5 = Planet('f', 93.8, 2459462.9, 3, 0, 0, 0.0001, 0.0089, 26.6, 3.8, 0, 0,)
    sys = System('TOI-1246', 5, star, [pl1, pl2, pl3, pl4, pl5])

    # Generate the setup file for the default system
    sys.make_setup_file(f'TestData/{sys.name}_st.csv', f"{sys.name}_default.py")
    # Run a Radvel fit on the default system setup file
    subprocess.run(["./radvel_bash.sh", f"{sys.name}_default.py", "nplanets"])




def calc_K(e,i,m1,m2,P):
    """
    Given system parameters, find radial velocity semi-amplitude
    """
    term1 = 203*(u.m/u.s)/np.sqrt(1-e**2)
    term2 = m2*np.sin(i)/c.M_jup
    term3 = ((m1/c.M_sun)+9.54*10**(-4)*(m2/c.M_jup))**(-2/3)
    term4 = (P/u.d)**(-1/3)

    return (term1*term2*term3*term4).to(u.m/u.s)

def sim_rvs(times, add_pl=True, per=0, t0=0, K=0, noise=0, filename=None):
    y_new_b = basic_model(times,18.6524,2458688.9403,1.4,0.56)
    y_new_c = basic_model(times,4.307412,2458686.5648,3.51,0.53)
    y_new_d = basic_model(times,5.9032,2458683.4746,3.45,0.53)
    y_new_e = basic_model(times,37.9198,2458700.7204,3.1,0.6)
    if add_pl:
        y_new_f = basic_model(times, per, t0, K, noise)
        y_new = y_new_b+y_new_c+y_new_d+y_new_e+y_new_f
        y_err = np.full((len(y_new),), 1.7).tolist()
    elif add_pl == False:
        y_new = y_new_b+y_new_c+y_new_d+y_new_e
        y_err = np.full((len(y_new),), 1.7).tolist()
    model = pd.DataFrame({'time':times, 'mnvel':y_new, 'errvel':y_err})
    model.to_csv(filename)
    return model