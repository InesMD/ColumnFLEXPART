from columnflexpart.classes.flexdatasetCO import FlexDatasetCO

fd = FlexDatasetCO('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/20191220_20191221/RELEASES_0_5/' , ct_dir='/work/bb1170/RUN/b382105/Data/CAMS/Concentration/regridded_1x1/', ct_name_dummy='something', chunks=dict(time=20, pointspec=4))
fd.load_measurement('/work/bb1170/RUN/b382105/Data/TCCON_wg_data/wg20080626_20200630.public.nc', 'TCCON')
tr = fd.trajectories
#tr.load_cams_data()
#tr.ct_endpoints(boundary = [110, 170, -45, -10])
tr.load_endpoints()
flux_path = '/work/bb1170/RUN/b382105/Data/CAMS/Fluxes/Regridded_1x1/'
#enhancement_inter = fd.enhancement(ct_file=args.flux_file, boundary=args.boundary, allow_read=args.read_only, interpolate=True)
enhancement = fd.enhancement(ct_file=flux_path, boundary=[110, 170, -45,-10], allow_read=True, interpolate=False)
'''
from columnflexpart.classes.flexdataset import FlexDataset
fd = FlexDataset('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/20191220_20191221/RELEASES_0_5/' , 
                 ct_dir='/work/bb1170/RUN/b382105/Data/CarbonTracker2022/Concentration/', 
                 ct_name_dummy='CT2022.molefrac_glb3x2_', chunks=dict(time=20, pointspec=4))
fd.load_measurement('/work/bb1170/RUN/b382105/Data/TCCON_wg_data/wg20080626_20200630.public.nc', 'TCCON')
tr = fd.trajectories
tr.load_ct_data()
tr.ct_endpoints(boundary = [110, 170, -45, -10])
'''