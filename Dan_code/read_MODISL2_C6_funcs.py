#from IPython import embed  #for debugging - put embed() in code for breakpoint


import math
import numpy as np

def get_hdf_data(SDS_NAME,hdf,disp_attr,fill_value) :


	#from pyhdf.SD import SD,SDC
	#import numpy as np

	#path='/group_workspaces/jasmin/asci/dgrosv/L2_C6/Hamish/'

	#file_hdf = 'MOD06_L2.A2016222.1135.006.2016222210127.hdf'

	# Will pass in hdf to the function - should be faster? And rest of these
	#hdf = SD(path + file_hdf , SDC.READ)
	#SDS_NAME  = 'Cloud_Top_Temperature'

	sds = hdf.select(SDS_NAME)
	dat = sds.get()
	dimsizes = dat.shape

	#function [dat,dimsizes,sampling_along,sampling_across,success]=get_hdf_data_dan(var_name,SD_id,INFO,unsigned,disp_attr)

	#success=0;

	##default to signed integers - should just be the QA arrays that are
	##unsigned
	#if nargin==3
	#    unsigned=0;
	#    disp_attr=0;
	#elseif nargin==4
	#    disp_attr=0;
	#end

	#nvar = hdfsd('nametoindex',SD_id,var_name);
	#sds_id = hdfsd('select',SD_id, nvar);
	#[name,rank,dimsizes,data_type,nattrs,status] = hdfsd('getinfo',sds_id);
	##dat = double(hdfread(INFO.Vgroup(1).Vgroup(2).SDS(nvar))); #this retrieves
	##the data
	    
	#start=zeros([1 length(dimsizes)]);
	#stride=ones([1 length(dimsizes)]);
	#edge=dimsizes;
	#[dat,status] = hdfsd('readdata',sds_id,start,stride,edge);

	#dat=double(dat); #convert to double as otherwise cannot manipulate the signed integers easily




	#[fill,status] = hdfsd('getfillvalue',sds_id);
	 
	#if length(data_type)==0 | prod(size(fill))==0
	#    disp('**** Read failure - likely the requested variable does not exist ***');       
	#    dat=NaN;
	#    dimsizes=NaN;
	#    sampling_along=NaN;
	#    sampling_across=NaN;    
	#    return
	#end


	#Get offset and scale factor attributes, and fill value
	offset = getattr(sds,'add_offset'); 
	sf = getattr(sds,'scale_factor')
	mask_value = sds.getfillvalue()

	ifill = (dat==mask_value) #create a separate array for this since otherwise it has trouble populating the integer array with np float NaNs

	dat = dat.astype(float) #IMPORTANT - convert to floats, or calculations will be wrong... but do it after masking logical
	dat[ifill]=fill_value


#	offset=offset.astype(float)
#	sf=sf.astype(float)


	#print 'embed read hdf'
	#embed()


	#offset=double(offset);
	#sf=double(sf);
	#fill=double(fill);

	##For some reason some data is listed as int8, which Matlab interprets as
	##being signed integers (that can be negative). This makes them unsigned.
	##if strcmp(data_type(1:3),'int')==1 & unsigned==1
	##    eval(['dat=unit' datatype(4:end) '(dat);';]);  
	## DO NOT USE uint8 - it just changes (saturates) negative values to zero

	#Nbits = str2num(data_type(4:end)); #e.g. gets the "16" from int16
	#modval = 2^Nbits; #e.g. 256 for 8 bits

	##e.g. signed 8 bit integers go from -128 to 127 and then wrap back again to
	##the start so if we add 256 to them all they should be unchanged. The -128
	##to -1 range maps to the 128 to 255 unsigned range. 
	##Add 256 to the negative values - the positive values remain unchanged as adding
	##256 to them would just bring them back to the same number - as they are
	##in int8 format in Matlab they saturate at 127 ( so int8(100)+256 = 127 )

	#ineg=find(dat<0);
	#dat(ineg)=dat(ineg)+modval;

	#end





	dat = dat - offset
	dat = dat * sf #apply offset and scale factor 
	#see modis-atmos.gsfc.nasa.gov/MOD08_D3/faq.html - offset needs to be
	    #SUBTRACTED and then the scale factor applied (multiply)

	    
	    #flip the last swath dimension - not sure if this is necessary - was
	    #something from Rob's script. Some arrays - such as re_diff have a 3rd
	    #dimension, so flip the dim with the largest size
	    #note dimsizes is not the same as size(dat)
	#    [maxval,dimflip]=max(size(dat));
	#    dat=flipdim(dat,dimflip);
	    
	    #get the sampling information - tells us which pixels were sampled -
	    #useful for lat and lon and other 5 km grid data
	     #iatts=7;
	     #[name,data_type,count,status] = hdfsd('attrinfo',sds_id,iatts);
	     #att_dat=hdfsd('readattr',sds_id,iatts);
	     #sampling_along = att_dat;

	     #iatts=8;
	     #[name,data_type,count,status] = hdfsd('attrinfo',sds_id,iatts);
	     #att_dat=hdfsd('readattr',sds_id,iatts);
	     #sampling_across = att_dat;

	#Not sure if still need these?? May need them for producing arrays from 5km variabiles (e.g. solar ZA) that match the 1km arrays.
	sampling_along = getattr(sds,'Cell_Along_Swath_Sampling')
	sampling_across = getattr(sds,'Cell_Across_Swath_Sampling')



	success=1;    
	    
	##  To display a list of attributes:-
	##disp_attr=0;
	#if disp_attr==1
	#    for iatts=0:nattrs-1
	#        [name,data_type,count,status] = hdfsd('attrinfo',sds_id,iatts);
	#        att_dat=hdfsd('readattr',sds_id,iatts);
	#        fprintf(1,'\n%d) %s = %s',iatts,name,num2str(att_dat));     
	#    end



	#end


	return (dat,dimsizes,sampling_along,sampling_across)

def getCDNC(cot,effradius,ctt): #Function to return cloud droplet number concentration (Nd) from cloud optical depth (cot) and effective radius (effradius, or re, in metres) and cloud top temperature (ctt). Only use for liquid pixles!

# Authours: Daniel Grosvenor & Hamish Gordon (U. Leeds)

	from thermo_functions import dqdz

	k = 0.8 # Brenguier et al 2011; Martin et al 1994 apparently suggests 0.67 and Painemal & Zuidema 0.88, all for MBL. Grosvenor & Wood (2014) uses 0.8
	f = 1.0  #Painemal & Zuidema aircraft obs show 0.7, but should set to 1.0 since get cancellation of errors with this and the re positive bias, and k value
	Q=2.0
	rho_w = 1000.0 #kg m^-3
	p = 850.0*100.0 #Pa - for use in the calc of the adiabatic rate of increase of qL with height (Cw). Assume constant since CTP from MODIS is fairly unreliable and Nd is not that sensitive to this anyway (see Grosvenor & Wood, 2014).

	Cw = dqdz(ctt, p)
	Nd_prefactor = (2.0*math.sqrt(10.0)/(k*math.pi*Q**3.0))

	Nd_sqrtarg_num = f*Cw*cot
	Nd_sqrtarg_den = rho_w*np.power(effradius,5.0) #Grosvenor & Wood; effradius is in metres here
	Nd = 1e-6*Nd_prefactor*np.sqrt(np.divide(Nd_sqrtarg_num,Nd_sqrtarg_den)) # units cm^-3

	LWP = 5.0/9.0*rho_w*cot*effradius #Cloud liquid water path
		#N.B. need to use 5.0, 9.0, etc. in Python or will give zeros...
	H = np.sqrt(2.0*LWP/Cw) #Cloud depth

	#embed()


	return (Nd,H,LWP,Cw)

# ------------------------------------------------------------------

def maskread_1km(mask1):

        #function  [mask_1km] = maskread_1km_Dan(mask1)
        #
        ##==================================================================================
        ## PROCEDURE    :       maskread_1km
        ## VERSION      :       1.0
        ## AUTHOR       :       Robert Wood - adapted for Matlab by Daniel Grosvenor 17th May, 2011. And for Python 27th Jan, 2017 - tested and produces the same output as Matlab
        ##                                                                 
        ## 
        ## DATE         :       March 19 2001
        ##
        ## DESCRIPTION  :       Takes HDF buffers for HDF variable ID 
        ##                      'Cloud_Mask_1km' (mask1) 
        ##                      and extracts mask into useable format
        ##===================================================================================
        ## BITS MASK
        #
        ##   bts=2^indgen(8)
        #
        ## CHECK DIMENSIONS of ARRAY
        ##   nx=n_elements(mask1(0,*,0))
        ##   ny=n_elements(mask1(0,0,*))
        #   
        #   nx=length(mask1(1,:,1));
        #   ny=length(mask1(1,1,:));

        mask1 = np.transpose(mask1,(2,1,0)) #Make order the same as for Matlab
        nx = mask1.shape[1]
        ny = mask1.shape[2]
#       print mask1.shape,nx,ny

        # Convert to integers
        mask1 = mask1.astype(int)

        #
        ## OUTPUTS
        #
        ##   mask_1km=bytarr(9,nx,ny)
        mask_1km=np.zeros((9,nx,ny), dtype=np.int)
         #        
        ##===========================================================================
        ## CLOUD MASK FLAGS for 1 km DATA
        ##  INFORMATION: (http://modis-atmos.gsfc.nasa.gov/reference_atbd.html  
        ##   
        ##  mask_1km(0)  :  CLOUD MASK 
        ##                      0: undetermined# 1: determined
        ##  mask_1km(1)  :  CLOUD MASK QUALITY FLAG
        ##                      0: 0-20# cloudy pixels# 1: 20-40## 3: 40-60## 4: 60-100#
        ##  mask_1km(2)  :  DAY/NIGHT FLAG
        ##                      0: night# 1: day
        ##  mask_1km(3)  :  SUN GLINT FLAG
        ##                      0: yes# 1: no
        ##  mask_1km(4)  :  SNOW/ICE FLAG
        ##                      0: yes# 1: no
        ##  mask_1km(5)  :  LAND/WATER FLAG
        ##                      0: water (ocean)# 1: coastal# 2: desert# 3: land
        ##  mask_1km(6)  :  HEAVY AEROSOL
        ##                      0: yes# 1: no
        ##  mask_1km(7)  :  THIN CIRRUS DETECTED
        ##                      0: yes# 1: no
        ##  mask_1km(8)  :  SHADOW FOUND
        ##                      0: yes# 1: no
        ##
        ##---------------------------------------------------------------------------   
        # 
        ##    bins=dec2bin(mask1,8); #dec2bin converts to an N by 8 array of the bits (in string format)
        ##   
        ##    
        ##    bins = reshape(bins,[size(mask1) 2]);
        ##    Lsiz = length(size(bins));
        ##    bins = permute(bins,[Lsiz 1:Lsiz-1]); #put the bits as the first dimension as makes life easier
        #   #becuase of way Matlab rearranges the dimensions when an array has singular dimensions
        #
        #
        #   #now we can use the bits using e.g. bin2dec_array(bins(2:4,1,:)); for
        #   #bits 2,3 and 4 (where 1 is the first bit) of the first byte of the 10
        #   
        #   
        ##    mask_1km(0,*,*)=byte(mask1(0,*,*) and bts(0))
        ##    mask_1km(1,*,*)=byte(((mask1(0,*,*) and bts(1))/bts(1))+$
        ##                    ((mask1(0,*,*) and bts(2))/bts(2))*2)
        ##    mask_1km(2,*,*)=byte((mask1(0,*,*) and bts(3))/bts(3))
        ##    mask_1km(3,*,*)=byte((mask1(0,*,*) and bts(4))/bts(4))
        ##    mask_1km(4,*,*)=byte((mask1(0,*,*) and bts(5))/bts(5))
        ##    mask_1km(5,*,*)=byte(((mask1(0,*,*) and bts(6))/bts(6))+$
        ##                    ((mask1(0,*,*) and bts(7))/bts(7))*2)  
        ##    mask_1km(6,*,*)=byte((mask1(1,*,*) and bts(0)))
        ##    mask_1km(7,*,*)=byte((mask1(1,*,*) and bts(1))/bts(1))
        ##    mask_1km(8,*,*)=byte((mask1(1,*,*) and bts(2))/bts(2))
        #
        #
        ##    mask_1km(1,:,:)  = bin2dec_array( bins(8,1,:,:) );
        ##    mask_1km(2,:,:)  = bin2dec_array( bins(6:7,1,:,:) );
        ##    mask_1km(3,:,:)  = bin2dec_array( bins(5,1,:,:) );
        ##    mask_1km(4,:,:)  = bin2dec_array( bins(4,1,:,:) );
        ##    mask_1km(5,:,:)  = bin2dec_array( bins(3,1,:,:) );
        ##    mask_1km(6,:,:)  = bin2dec_array( bins(1:2,1,:,:) );
        ##    mask_1km(7,:,:)  = bin2dec_array( bins(8,2,:,:) );
        ##    mask_1km(8,:,:)  = bin2dec_array( bins(7,2,:,:) );
        ##    mask_1km(9,:,:)  = bin2dec_array( bins(6,2,:,:) );
        #

#Use this to convert from double to integers :-
#x=x.astype(int)
#And this seems to work ok for bitand and bitshifting as for Matlab (without worrying about unsigned, etc.)
#a = (x >> 6) & 3  #to bitshift by -6 and bitand with 3 (as for bitand(bitshift(mask1(1,:,:),-6),3) in Matlab)
#double check on whole array 

        mask_1km[0,:,:]  = ( mask1[0,:,:] >> 0 ) & 1
        mask_1km[1,:,:]  = ( mask1[0,:,:] >> 1 ) & 3
        mask_1km[2,:,:]  = ( mask1[0,:,:] >> 3 ) & 1
        mask_1km[3,:,:]  = ( mask1[0,:,:] >> 4 ) & 1
        mask_1km[4,:,:]  = ( mask1[0,:,:] >> 5 ) & 1
        mask_1km[5,:,:]  = ( mask1[0,:,:] >> 6 ) & 3
        mask_1km[6,:,:]  = ( mask1[1,:,:] >> 0 ) & 1
        mask_1km[7,:,:]  = ( mask1[1,:,:] >> 1 ) & 1
        mask_1km[8,:,:]  = ( mask1[1,:,:] >> 2 ) & 1

# Matlab version :-
        #   mask_1km(1,:,:)  = bitand(bitshift(mask1(1,:,:), 0),1);
        #   mask_1km(2,:,:)  = bitand(bitshift(mask1(1,:,:),-1),3);   
        #   mask_1km(3,:,:)  = bitand(bitshift(mask1(1,:,:),-3),1);      
        #   mask_1km(4,:,:)  = bitand(bitshift(mask1(1,:,:),-4),1);   
        #   mask_1km(5,:,:)  = bitand(bitshift(mask1(1,:,:),-5),1);      
        #   mask_1km(6,:,:)  = bitand(bitshift(mask1(1,:,:),-6),3);   
        #   mask_1km(7,:,:)  = bitand(bitshift(mask1(2,:,:), 0),1);      
        #   mask_1km(8,:,:)  = bitand(bitshift(mask1(2,:,:),-1),1);   
        #   mask_1km(9,:,:)  = bitand(bitshift(mask1(2,:,:),-2),1);      
        #
        #
        #
        #end
        #
        #
        #
        #

        mask_1km = np.transpose(mask_1km,(2,1,0)) #Transpose back to be consistent with other arrays

        return (mask_1km)

