import numpy as np

def Mars_2000(lonlat_range=180):
    coord_sys = ('GEOGCS["Mars 2000",'
                 'DATUM["D_Mars_2000",'
                 'SPHEROID["Mars_2000_IAU_IAG",3396190.0,169.89444722361179]],'
                 'PRIMEM["Greenwich",0],'
                 'UNIT["Decimal_Degree",0.0174532925199433]],')
    if lonlat_range == 180:
        coord_sys = coord_sys + 'USAGE[SCOPE["unknown"],AREA["World"],BBOX[-90,-180,90,180]]'
    elif lonlat_range == 360:
        coord_sys = coord_sys + 'USAGE[SCOPE["unknown"],AREA["World"],BBOX[-90,0,90,360]]'
    return(coord_sys)

def Moon_2000(lonlat_range=180):

    coord_sys = ('GEOGCRS["GCS_Moon_2000",'
                 'DATUM["D_Moon_2000",'
                 'ELLIPSOID["Moon_2000_IAU_IAG",1737400,0,'
                 'LENGTHUNIT["metre",1]]],'
                 'PRIMEM["Reference_Meridian",0,'
                 'ANGLEUNIT["degree",0.0174532925199433]],'
                 'CS[ellipsoidal,2],'
                 'AXIS["geodetic latitude (Lat)",north,'
                 'ORDER[1],'
                 'ANGLEUNIT["degree",0.0174532925199433]],'
                 'AXIS["geodetic longitude (Lon)",east,'
                 'ORDER[2],'
                 'ANGLEUNIT["degree",0.0174532925199433]],'
                 'USAGE[SCOPE["unknown"],AREA["World"], BBOX[-90,-180,90,180]],'
                 'ID["ESRI",104903]]')

    if lonlat_range == 360:
        coord_sys.replace("BBOX[-90,-180,90,180]],", "BBOX[-90,0,90,360]],")
    return(coord_sys)

def Moon_Equirectangular():
    proj = ('PROJCS["Equirectangular Moon",'
            'GEOGCS["GCS_Moon",DATUM["D_Moon",'
            'SPHEROID["Moon_localRadius",1737400,0]],'
            'PRIMEM["Reference_Meridian",0],'
            'UNIT["degree",0.0174532925199433,'
            'AUTHORITY["EPSG","9122"]]],'
            'PROJECTION["Equirectangular"],'
            'PARAMETER["standard_parallel_1",0],'
            'PARAMETER["central_meridian",0],'
            'PARAMETER["false_easting",0],'
            'PARAMETER["false_northing",0],'
            'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
            'AXIS["Easting",EAST],'
            'AXIS["Northing",NORTH]]')
    return (proj)

def Moon_Equirectangular_360():
    proj = ('PROJCS["Equirectangular Moon",'
            'GEOGCS["GCS_Moon",DATUM["D_Moon",'
            'SPHEROID["Moon_localRadius",1737400,0]],'
            'PRIMEM["Reference_Meridian",0],'
            'UNIT["degree",0.0174532925199433,'
            'AUTHORITY["EPSG","9122"]]],'
            'PROJECTION["Equirectangular"],'
            'PARAMETER["standard_parallel_1",0],'
            'PARAMETER["central_meridian",180],'
            'PARAMETER["false_easting",0],'
            'PARAMETER["false_northing",0],'
            'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
            'AXIS["Easting",EAST],'
            'AXIS["Northing",NORTH]]')
    return (proj)

def Moon_Equidistant_Cylindrical():
    proj = ('PROJCS["Moon_Equidistant_Cylindrical",'
            'GEOGCS["Moon 2000",DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Decimal_Degree",0.0174532925199433]],'
            'PROJECTION["Equidistant_Cylindrical"],'
            'PARAMETER["False_Easting",0],'
            'PARAMETER["False_Northing",0],'
            'PARAMETER["Central_Meridian",0],'
            'PARAMETER["Standard_Parallel_1",0],'
            'UNIT["Meter",1]]')

    return (proj)

def Moon_Mollweide(longitude):
    proj = ('PROJCS["Moon_Mollweide",'
            'GEOGCS["Moon 2000",'
            'DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Decimal_Degree",0.0174532925199433]],'
            'PROJECTION["Mollweide"],'
            'PARAMETER["False_Easting",0],'
            'PARAMETER["False_Northing",0],'
            'PARAMETER["Central_Meridian",0],'
            'UNIT["Meter",1]]')

    proj = proj.replace('_Meridian",0', '_Meridian",' + str(int(longitude)))

    return (proj)

def Moon_Mercator():

    proj = ('PROJCS["Moon_Mercator_AUTO",'
            'GEOGCS["Moon 2000",'
            'DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Decimal_Degree",0.0174532925199433]],'
            'PROJECTION["Mercator"],'
            'PARAMETER["False_Easting",0],'
            'PARAMETER["False_Northing",0],'
            'PARAMETER["Central_Meridian",0],'
            'PARAMETER["Standard_Parallel_1",0],'
            'UNIT["Meter",1]]')

    return (proj)

def Moon_Lambert_Conformal_Conic_N(longitude):
    proj = ('PROJCS["Moon_Lambert_Conformal_Conic",'
            'GEOGCS["Moon 2000",'
            'DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Decimal_Degree",0.0174532925199433]],'
            'PROJECTION["Lambert_Conformal_Conic"],'
            'PARAMETER["False_Easting",0],'
            'PARAMETER["False_Northing",0],'
            'PARAMETER["Central_Meridian",0],'
            'PARAMETER["Standard_Parallel_1",30],'
            'PARAMETER["Standard_Parallel_2",60],'
            'PARAMETER["Latitude_Of_Origin",45],'
            'UNIT["Meter",1]]')

    proj = proj.replace('_Meridian",0', '_Meridian",' + str(int(longitude)))
    #proj = proj.replace('_Meridian",0', '_Meridian",' + str(int(round(longitude))))

    return(proj)


def Moon_Lambert_Conformal_Conic_S(longitude):
    proj = ('PROJCS["Moon_Lambert_Conformal_Conic",'
            'GEOGCS["Moon 2000",'
            'DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Decimal_Degree",0.0174532925199433]],'
            'PROJECTION["Lambert_Conformal_Conic"],'
            'PARAMETER["False_Easting",0],'
            'PARAMETER["False_Northing",0],'
            'PARAMETER["Central_Meridian",0],'
            'PARAMETER["Standard_Parallel_1",-60],'
            'PARAMETER["Standard_Parallel_2",-30],'
            'PARAMETER["Latitude_Of_Origin",-45],'
            'UNIT["Meter",1]]')

    proj = proj.replace('_Meridian",0', '_Meridian",' + str(int(longitude)))
    #proj = proj.replace('_Meridian",0', '_Meridian",' + str(int(round(longitude))))

    return (proj)

def Moon_North_Pole_Stereographic():
    proj = ('PROJCS["Moon_North_Pole_Stereographic",'
            'GEOGCS["Moon 2000",'
            'DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Decimal_Degree",0.0174532925199433]],'
            'PROJECTION["Stereographic"],'
            'PARAMETER["False_Easting",0],'
            'PARAMETER["False_Northing",0],'
            'PARAMETER["Central_Meridian",0],'
            'PARAMETER["Scale_Factor",1],'
            'PARAMETER["Latitude_Of_Origin",90],'
            'UNIT["Meter",1]]')
    return (proj)

def Moon_South_Pole_Stereographic():
    proj = ('PROJCS["Moon_South_Pole_Stereographic",'
            'GEOGCS["Moon 2000",'
            'DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Decimal_Degree",0.0174532925199433]],'
            'PROJECTION["Stereographic"],'
            'PARAMETER["False_Easting",0],'
            'PARAMETER["False_Northing",0],'
            'PARAMETER["Central_Meridian",0],'
            'PARAMETER["Scale_Factor",1],'
            'PARAMETER["Latitude_Of_Origin",-90],'
            'UNIT["Meter",1]]')
    return (proj)

def mollweide_proj(a, b):
    default_mol = ('+proj=moll +lon_0=0 +x_0=0 +y_0=0 +a=1737400 +b=1737400'
                   ' +units=m +no_defs')

    proj = default_mol.replace('+a=1737400', '+a=' + str(int(a)))
    proj = proj.replace('+b=1737400', '+b=' + str(int(b)))

    return(proj)

def stereographic_npole(a,b):
    default_stereog = ('+proj=stere +lat_0=90 +lon_0=0 +k=1 +x_0=0 +y_0=0 '
                       '+a=1737400 +b=1737400 +units=m +no_defs')

    proj = default_stereog.replace('+a=1737400', '+a=' + str(int(a)))
    proj = proj.replace('+b=1737400', '+b=' + str(int(b)))

    return (proj)

def stereographic_spole(a,b):
    default_stereog = ('+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 '
                       '+a=1737400 +b=1737400 +units=m +no_defs')

    proj = default_stereog.replace('+a=1737400', '+a=' + str(int(a)))
    proj = proj.replace('+b=1737400', '+b=' + str(int(b)))

    return (proj)

def equirectangular_proj(longitude, latitude, a, b, default=True):

    if default:
        longitude = 0
        latitude = 0

    default_eqc = ('+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0'
                   ' +a=1737400 +b=1737400 +units=m +no_defs')

    proj = default_eqc.replace('+lon_0=0','+lon_0=' + str(int(longitude)))
    proj = proj.replace('+lat_ts=0','+lat_ts=' + str(int(latitude)))
    proj = proj.replace('+a=1737400', '+a=' + str(int(a)))
    proj = proj.replace('+b=1737400', '+b=' + str(int(b)))

    return(proj)

def select_proj(longitude, latitude):

    # latitude = np.round(latitude,decimals=1)
    if np.logical_and(latitude >= -30.0, latitude <= 30.0):
        proj = Moon_Equidistant_Cylindrical()
    elif np.logical_and(latitude < -30.0, latitude >= -60.0):
        proj = Moon_Lambert_Conformal_Conic_S(longitude)
    elif np.logical_and(latitude > 30.0, latitude <= 60.0):
        proj = Moon_Lambert_Conformal_Conic_N(longitude)
    elif latitude > 60.0:
        proj = Moon_North_Pole_Stereographic()
    elif latitude < -60.0:
        proj = Moon_South_Pole_Stereographic()

    return proj


