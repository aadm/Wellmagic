import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from las import LASReader

#===== DEFINIZIONE COLORI PER LEGENDA CLASSI
# litho-fluid classes: 0=undef, 1=brine sand, 2=oil sand, 3=shale,
import matplotlib.patches as mpatches
legend_lfc0 = mpatches.Patch(color='#B3B3B3', label='0')  # 0=undef
legend_lfc1 = mpatches.Patch(color='blue',    label='1')  # 1=brine sand
legend_lfc2 = mpatches.Patch(color='green',   label='2')  # 2=oil sand
legend_lfc3 = mpatches.Patch(color='red',     label='3')  # 3=gas sand
legend_lfc4 = mpatches.Patch(color='#996633', label='4')  # 4=shale


#===== creazione colormap facies
import matplotlib.colors as colors
#      0=undef   1=bri  2=oil   3=gas 4=shale
ccc = ['#B3B3B3','blue','green','red','#996633',]
cmap_facies = colors.ListedColormap(ccc[0:len(ccc)], 'indexed')


# funzione voigt-reuss-hill
def vrh(volumes,k,mu):
    f=np.array(volumes).T
    k=np.resize(np.array(k),np.shape(f))
    mu=np.resize(np.array(mu),np.shape(f))

    k_u = np.sum(f*k,axis=1)
    k_l = 1./np.sum(f/k,axis=1)
    mu_u = np.sum(f*mu,axis=1)
    mu_l = 1./np.sum(f/mu,axis=1)
    k0 = (k_u+k_l)/2.
    mu0 = (mu_u+mu_l)/2.
    return [k_u, k_l, mu_u, mu_l, k0, mu0]
# def vrh(volumes,k,mu):
#     f=np.array(volumes)
#     k=np.resize(np.array(k),np.shape(f))
#     mu=np.resize(np.array(mu),np.shape(f))
#     k_u = np.sum(f*k)
#     k_l = 1/np.sum(f/k)
#     mu_u = np.sum(f*mu)
#     mu_l = 1/np.sum(f/mu)
#     k0 = (k_u+k_l)/2
#     mu0 = (mu_u+mu_l)/2
#     return [k_u, k_l, mu_u, mu_l, k0, mu0]

# funzione gassmann
def frm(vp1, vs1, rho1, rho_fl1, k_fl1, rho_fl2, k_fl2, k0, phi):
    vp1=vp1/1000
    vs1=vs1/1000
    rho2 = rho1-phi*rho_fl1+phi*rho_fl2
    mu1 = rho1*vs1**2.
    k1 = rho1*vp1**2-(4./3.)*mu1
    kdry= (k1 * ((phi*k0)/k_fl1+1-phi)-k0) / ((phi*k0)/k_fl1+(k1/k0)-1-phi)
    k2 = kdry + (1- (kdry/k0))**2 / ( (phi/k_fl2) + ((1-phi)/k0) - (kdry/k0**2) )
    mu2 = mu1
    vp2 = np.sqrt(((k2+(4./3)*mu2))/rho2)
    vs2 = np.sqrt((mu2/rho2))
    return [vp2*1000, vs2*1000, rho2, k2]

# all'inizio avevo scelto di usare uno dei pozzi Cinguvu usati nel lavoro fatto per A.Castoro e M.Fervari,
# vedi cinguvu_dataprep.py, cinguvu_plots.py e cinguvu_Montecarlo.py
# Poi ho deciso di usare un dataset non proprietario e ho preso il Well 2 di Quantitative Seismic Interpretation by Avseth et al.,
# vedi quindi qsi_dataprep.py per come ho preparato il file, con merge e creazione di log petrofisici.


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# tolgo il log di facies dal csv preparato con qsi_dataprep.py perche' ho deciso
# di rifarla qui dentro (cioe' nel notebook) la creazione del log di facies.
#l=pd.read_csv('qsiwell2.csv')
#l=l.drop(['LFC'],axis=1)
#l.to_csv('qsiwell2.csv',index=False)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


l=pd.read_csv('qsiwell2.csv')
# l=l.ix[(l.DEPTH>=2100) & (l.DEPTH<=2400)]

#=== CREAZIONE LOG DI FACIES
sand_cutoff=0.2
ssb=((l.VSH<=sand_cutoff) & (l.SW>=0.9))  # brine sand
sso=((l.VSH<=sand_cutoff) & (l.SW<0.9))   # hc sand (insitu=oil)
sh=(l.VSH>sand_cutoff)                    # shales
print("sst/res=%d, brine sst=%d, oil sst=%d, shale=%d" % ((np.count_nonzero(ssb)+np.count_nonzero(sso)),np.count_nonzero(ssb),np.count_nonzero(sso),np.count_nonzero(sh)))

# litho-fluid classes: 0=undef, 1=brine sand, 2=oil sand, 3=gas sand, 4=shale
lfc=np.zeros(np.shape(l.VSH))
lfc[ssb.values]=1
lfc[sso.values]=2
lfc[sh.values]=4
l['LFC']= lfc

#===== LOG PLOT + FACIES PLOT
ztop=2100; zbot=2250
l['dummy']=np.zeros(len(l.VSH))

f, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(8, 6))
ax[0].plot(l.VSH, l.DEPTH, '-g', label='Vsh')
ax[0].plot(l.SW, l.DEPTH, '-b', label='Sw')
ax[0].plot(l.PHI, l.DEPTH, '-k', label='phi')
ax[1].plot(l.IP, l.DEPTH, '-', color='0.5')
ax[2].plot(l.VPVS, l.DEPTH, '-', color='0.5')
#~~~~ plot facies log su una unica colonna (poco leggibile)
# ax[3].plot(l.dummy[l.LFC==0],l.DEPTH[l.LFC==0],'s',markeredgecolor='#B3B3B3',label='LFC 0',markersize=2)
# ax[3].plot(l.dummy[l.LFC==1],l.DEPTH[l.LFC==1],'s',markeredgecolor='blue',   label='LFC 1',markersize=2)
# ax[3].plot(l.dummy[l.LFC==2],l.DEPTH[l.LFC==2],'s',markeredgecolor='green',  label='LFC 2',markersize=2)
# ax[3].plot(l.dummy[l.LFC==4],l.DEPTH[l.LFC==4],'s',markeredgecolor='#996633',label='LFC 4',markersize=2)
# ax[3].set_xlabel('LFC'),           ax[3].set_xlim(-1,1)
#~~~~ plot facies log su una quattro sottocolonne separate
ax[3].plot(l.LFC[l.LFC==0],l.DEPTH[l.LFC==0],'s',color='#B3B3B3',markeredgecolor='#B3B3B3',label='LFC 0',markersize=2)
ax[3].plot(l.LFC[l.LFC==1],l.DEPTH[l.LFC==1],'s',color='blue',   markeredgecolor='blue',   label='LFC 1',markersize=2)
ax[3].plot(l.LFC[l.LFC==2],l.DEPTH[l.LFC==2],'s',color='green',  markeredgecolor='green',  label='LFC 2',markersize=2)
ax[3].plot(l.LFC[l.LFC==4],l.DEPTH[l.LFC==4],'s',color='#996633',markeredgecolor='#996633',label='LFC 4',markersize=2)
ax[1].set_xlabel("Ip [m/s*g/cc]"), ax[1].set_xlim(3000,9000)
ax[2].set_xlabel("Vp/Vs"),         ax[2].set_xlim(1.5,3)
ax[3].set_xlabel('LFC'),           ax[3].set_xlim(-1,5)
ax[0].legend(fontsize='small', loc='lower right')
# ax[3].legend(fontsize='small', loc='lower right', handles=[legend_lfc0, legend_lfc1, legend_lfc2, legend_lfc3, legend_lfc4])
ax[3].set_xticklabels([])
ax[3].xaxis.grid(False)
# ax[0].set_ylim(ztop,zbot), ax[0].set_xlim(-.1,1.1)
ax[0].invert_yaxis()
for i in range(len(ax)):
    ax[i].locator_params(axis='x', nbins=4)
    ax[i].grid()

#~~~~ versione migliorata, da usare nel notebook, usa imshow per mostrare la colonnina con le facies
ztop=2030; zbot=2410

ll=l.ix[(l.DEPTH>=ztop) & (l.DEPTH<=zbot)]
cluster=np.repeat(np.expand_dims(ll['LFC'].values,1),100,1)

f, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 6))
ax[0].plot(ll.VSH, ll.DEPTH, '-g', label='Vcl')
ax[0].plot(ll.SW, ll.DEPTH, '-b', label='Sw')
ax[0].plot(ll.PHI, ll.DEPTH, '-k', label='phi')
ax[1].plot(ll.IP, ll.DEPTH, '-', color='0.5')
ax[2].plot(ll.VPVS, ll.DEPTH, '-', color='0.5')
im=ax[3].imshow(cluster, interpolation='none', aspect='auto',cmap=cmap_facies,vmin=0,vmax=4)
cbar=plt.colorbar(im, ax=ax[3])
cbar.set_label('0=undef,1=brine,2=oil,3=gas,4=shale')
cbar.set_ticks(range(0,4+1)), cbar.set_ticklabels(range(0,4+1))
for i in range(len(ax)-1):
    ax[i].set_ylim(ztop,zbot)
    ax[i].invert_yaxis()
    ax[i].grid()
    ax[i].locator_params(axis='x', nbins=4)
ax[0].set_xlabel("Vcl/phi/Sw"),    ax[0].set_xlim(-.1,1.1), ax[0].legend(fontsize='small', loc='lower right')
ax[1].set_xlabel("Ip [m/s*g/cc]"), ax[1].set_xlim(3000,9000)
ax[2].set_xlabel("Vp/Vs"),         ax[2].set_xlim(1.5,3)
ax[3].set_xlabel('LFC')
ax[1].set_yticklabels([]), ax[2].set_yticklabels([]), ax[3].set_yticklabels([]), ax[3].set_xticklabels([])


#===== FLUID REPLACEMENT

# just to check I used flprop to calculate oil elastic properties
# using average press and temp values for a depth of ~2000m in 10m water depth.

# ou=aa.flprop(0, 80000, 32, 0.6, 64, 0, 0, 20, 77, 1, 0)
#                   |                        |   |
#                   +--brine salinity, doesnt matter for oil
#                                            |   |
#                                            |   +--temp
#                                            +--press
# ou[6]
# Out[259]: 1.097  vp oil

# ou[7]
# Out[260]: 0.781 rho oil

# ou[8]
# Out[261]:  0.941 k oil


# # rho_matrix = vol_min1*rho_min1 + vol_min2*rho_min2
# w2['RHOm']=w2.VCL*rho_clay + (1-w2.VCL)*rho_qtz

# # rho_fluid = Sw*rho_water + (1-Sw)*rho_oil
# w2['RHOf']=w2.SW*rho_w + (1-w2.SW)*rho_o

# input elastic parameters; see also QSI, p.261, 336, 338
rho_qz=2.65;  k_qz=37;  mu_qz=44
rho_sh=2.81;  k_sh=15;  mu_sh=5
rho_b=1.09;   k_b=2.8
rho_o=0.78;   k_o=0.94 # # oil gravity: 32 API, GOR:    64
rho_g=0.25;   k_g=0.06              # fluid properties, gas

# calcolo k_matrix
shale=l.VSH
sand=1-shale-l.PHI
shaleN = shale/(shale+sand)  # normalized shale and sand volumes
sandN  = sand/(shale+sand)
[k_u, k_l, mu_u, mu_l, k0, mu0]=vrh([shaleN, sandN],[k_sh,k_qz],[mu_sh,mu_qz])

[k_u, k_l, mu_u, mu_l, k0, mu0]=vrh([l.VSH, 1-l.VSH],[k_sh,k_qz],[mu_sh,mu_qz])
# calcolo k_fluid
[tmp, k_fl, tmp, tmp, tmp, tmp]=vrh([l.SW, 1-l.SW],[k_b,k_o],[0,0])
# calcolo rho_fluid
rho_fl=l.SW*rho_b + (1-l.SW)*rho_o


# ro2 = ro1-phi*rofl1+phi*rofl2
# mu1 = ro1*vs1**2.
# k1 = ro1*vp1**2-(4./3.)*mu1
# a = k1/(k0-k1)-kfl1/(phi*(k0-kfl1))+kfl2/(phi*(k0-kfl2))
# k2 = k0*a/(1.+a)
# mu2 = mu1
# vp2 = np.sqrt(((k2+(4./3)*mu2))/ro2)
# vs2 = np.sqrt((mu2/ro2))

[vpb, vsb, rhob, kb]=frm(l.VP, l.VS, l.RHO, rho_fl, k_fl, rho_b, k_b, k0, l.PHI)
[vpo, vso, rhoo, ko]=frm(l.VP, l.VS, l.RHO, rho_fl, k_fl, rho_o, k_o, k0, l.PHI)
[vpg, vsg, rhog, kg]=frm(l.VP, l.VS, l.RHO, rho_fl, k_fl, rho_g, k_g, k0, l.PHI)

plt.figure(figsize=(5,10))
plt.plot(vpb, l.DEPTH, '-b')
plt.plot(vpo, l.DEPTH, '-g')
plt.plot(l.VP, l.DEPTH, '-', color='0.5')
plt.xlim(1000,6000), plt.ylim(2100,2300)
plt.gca().invert_yaxis()

# fluid replacement TO BRINE on oil sands, vp2[l.LFC==2]
l['VP_FRMB']=l.VP
l['VS_FRMB']=l.VS
l['RHO_FRMB']=l.RHO
l['VP_FRMB'][ssb|sso]=vpb[ssb|sso]
l['VS_FRMB'][ssb|sso]=vsb[ssb|sso]
l['RHO_FRMB'][ssb|sso]=rhob[ssb|sso]
l['IP_FRMB']=l.VP_FRMB*l.RHO_FRMB
l['IS_FRMB']=l.VS_FRMB*l.RHO_FRMB
l['VPVS_FRMB']=l.VP_FRMB/l.VS_FRMB

l['VP_FRMO']=l.VP
l['VS_FRMO']=l.VS
l['RHO_FRMO']=l.RHO
l['VP_FRMO'][ssb|sso]=vpo[ssb|sso]
l['VS_FRMO'][ssb|sso]=vso[ssb|sso]
l['RHO_FRMO'][ssb|sso]=rhoo[ssb|sso]
l['IP_FRMO']=l.VP_FRMO*l.RHO_FRMO
l['IS_FRMO']=l.VS_FRMO*l.RHO_FRMO
l['VPVS_FRMO']=l.VP_FRMO/l.VS_FRMO

l['VP_FRMG']=l.VP
l['VS_FRMG']=l.VS
l['RHO_FRMG']=l.RHO
l['VP_FRMG'][ssb|sso]=vpg[ssb|sso]
l['VS_FRMG'][ssb|sso]=vsg[ssb|sso]
l['RHO_FRMG'][ssb|sso]=rhog[ssb|sso]
l['IP_FRMG']=l.VP_FRMG*l.RHO_FRMG
l['IS_FRMG']=l.VS_FRMG*l.RHO_FRMG
l['VPVS_FRMG']=l.VP_FRMG/l.VS_FRMG

# aggiornamento facies log

# litho-fluid classes: 0=undef, 1=brine sand, 2=oil sand, 3=gas sand, 4=shale,
temp_lfc_b=np.zeros(np.shape(l.VSH))
temp_lfc_b[ssb.values | sso.values]=1
temp_lfc_b[sh.values]=4
l['LFC_B']= temp_lfc_b

temp_lfc_o=np.zeros(np.shape(l.VSH))
temp_lfc_o[ssb.values | sso.values]=2
temp_lfc_o[sh.values]=4
l['LFC_O']= temp_lfc_o

temp_lfc_g=np.zeros(np.shape(l.VSH))
temp_lfc_g[ssb.values | sso.values]=3
temp_lfc_g[sh.values]=4
l['LFC_G']= temp_lfc_g

# quick qc frm
plt.figure(figsize=(5,10))
plt.plot(l.VP_FRMB, l.DEPTH, '-b')
plt.plot(l.VP_FRMO, l.DEPTH, '-g')
plt.plot(l.VP_FRMG, l.DEPTH, '-r')
plt.plot(l.VP, l.DEPTH, '-k')
plt.xlim(1000,6000), plt.ylim(2100,2300)
plt.gca().invert_yaxis()

f, ax = plt.subplots(nrows=1, ncols=4, sharey=True, sharex=True, figsize=(16, 4))
ax[0].scatter(l.IP,l.VPVS,20,l.LFC,marker='o',edgecolors='none',alpha=0.5,cmap=cmap_facies,vmin=0,vmax=4)
ax[1].scatter(l.IP_FRMB,l.VPVS_FRMB,20,l.LFC_B,marker='o',edgecolors='none',alpha=0.5,cmap=cmap_facies,vmin=0,vmax=4)
ax[2].scatter(l.IP_FRMO,l.VPVS_FRMO,20,l.LFC_O,marker='o',edgecolors='none',alpha=0.5,cmap=cmap_facies,vmin=0,vmax=4)
ax[3].scatter(l.IP_FRMG,l.VPVS_FRMG,20,l.LFC_G,marker='o',edgecolors='none',alpha=0.5,cmap=cmap_facies,vmin=0,vmax=4)
ax[0].set_xlim(3000,9000); ax[0].set_ylim(1.5,3);
ax[0].set_title('original data');
ax[1].set_title('FRM to brine');
ax[2].set_title('FRM to oil');
ax[3].set_title('FRM to gas');
for i in range(len(ax)): ax[i].grid()


#****************************************************************
# l.to_csv('qsiwell2_frm.csv',index=False)  # temporary output
l=pd.read_csv('qsiwell2_frm.csv')
#****************************************************************

#===== LOG PLOT + FACIES PLOT !!!FRM!!!
ztop=2150; zbot=2200
ll=l.ix[(l.DEPTH>=ztop) & (l.DEPTH<=zbot)]

cluster=np.repeat(np.expand_dims(ll['LFC'].values,1),100,1)

f, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 6))
ax[0].plot(ll.VSH, ll.DEPTH, '-g', label='Vsh')
ax[0].plot(ll.SW, ll.DEPTH, '-b', label='Sw')
ax[0].plot(ll.PHI, ll.DEPTH, '-k', label='phi')
ax[1].plot(ll.IP_FRMG, ll.DEPTH, '-r')
ax[1].plot(ll.IP_FRMB, ll.DEPTH, '-b')
ax[1].plot(ll.IP, ll.DEPTH, '-', color='0.5')
ax[2].plot(ll.VPVS_FRMG, ll.DEPTH, '-r')
ax[2].plot(ll.VPVS_FRMB, ll.DEPTH, '-b')
ax[2].plot(ll.VPVS, ll.DEPTH, '-', color='0.5')
im=ax[3].imshow(cluster, interpolation='none', aspect='auto',cmap=cmap_facies,vmin=0,vmax=4)
cbar=plt.colorbar(im, ax=ax[3])
cbar.set_label('0=undef,1=brine,2=oil,3=gas,4=shale')
cbar.set_ticks(range(0,4+1)); 
for i in range(len(ax)-1):
    ax[i].set_ylim(ztop,zbot)
    ax[i].invert_yaxis()
    ax[i].grid()
    ax[i].locator_params(axis='x', nbins=4)
ax[0].legend(fontsize='small', loc='lower right')
ax[0].set_xlabel("Vcl/phi/Sw"),    ax[0].set_xlim(-.1,1.1)
ax[1].set_xlabel("Ip [m/s*g/cc]"), ax[1].set_xlim(3000,9000)
ax[2].set_xlabel("Vp/Vs"),         ax[2].set_xlim(1.5,3)
ax[3].set_xlabel('LFC')
ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([]); ax[3].set_xticklabels([]);


#===== SCATTER PLOTS

#f, ax = plt.subplots(1)
#ax.plot(l.IP[l.LFC==1],l.VPVS[l.LFC==1],'ob',markersize=4,markeredgewidth=0,label='brine sand')
#ax.plot(l.IP[l.LFC==2],l.VPVS[l.LFC==2],'og',markersize=4,markeredgewidth=0,label='oil sand')
#ax.plot(l.IP[l.LFC==3],l.VPVS[l.LFC==3],'or', markersize=4,markeredgewidth=0,label='gas sand',)
#ax.plot(l.IP[l.LFC==4],l.VPVS[l.LFC==4],'o', markersize=4,markeredgewidth=0,label='shale',color='#996633')
## additional frm datapoints
#ax.plot(l.IP_FRMB[l.LFC_B==1],l.VPVS_FRMB[l.LFC_B==1],'^b',markersize=5,markeredgewidth=0,label='brine sand (FRM)')
#ax.plot(l.IP_FRMO[l.LFC_O==2],l.VPVS_FRMO[l.LFC_O==2],'^g',markersize=5,markeredgewidth=0,label='oil sand (FRM)')
#ax.plot(l.IP_FRMG[l.LFC_G==3],l.VPVS_FRMG[l.LFC_G==3],'^r',markersize=5,markeredgewidth=0,label='gas sand (FRM)')
#plt.xlabel('Ip [m/s*g/cc]'); plt.ylabel('Vp/Vs')
#ax.legend(fontsize='small', loc='lower right', handles=[legend_lfc1, legend_lfc2, legend_lfc3, legend_lfc4])
## ax.legend(fontsize='small', loc='lower right')
#plt.xlim(3000,9000); plt.ylim(1.5,3)
#plt.grid()

plt.scatter(l.IP,l.VPVS,20,l.LFC,marker='o',edgecolors='none',alpha=0.7,cmap=cmap_facies,vmin=0,vmax=4)
plt.xlim(3000,9000); plt.ylim(1.5,3); plt.grid()
cbar=plt.colorbar()
cbar.set_label('0=undef,1=brine,2=oil,3=gas,4=shale')
cbar.set_ticks(range(0,4+1)); cbar.set_ticklabels(range(0,4+1));


#===== STATISTICAL ANALYSIS: data preparation
l=l.ix[(l.DEPTH>=2100) & (l.DEPTH<=2400)]

lognames0=['LFC','IP','VPVS']
lognames1=['LFC_B','IP_FRMB', 'VPVS_FRMB']
lognames2=['LFC_O','IP_FRMO', 'VPVS_FRMO']
lognames3=['LFC_G','IP_FRMG', 'VPVS_FRMG']
ww0=l[pd.notnull(l.LFC)].ix[:,lognames0];
ww1=l[pd.notnull(l.LFC)].ix[:,lognames1];  ww1.columns=[lognames0]
ww2=l[pd.notnull(l.LFC)].ix[:,lognames2];  ww2.columns=[lognames0]
ww3=l[pd.notnull(l.LFC)].ix[:,lognames3];  ww3.columns=[lognames0]
ww=pd.concat([ww0, ww1, ww2, ww3])

nlfc=int(ww.LFC.max())
nlogs=len(ww.columns)-1
names_mean=['']*nlogs
names_cov=['']*nlogs*nlogs
for i,aa in enumerate(names_mean):
    names_mean[i]='mean'+str(i)
for i,aa in enumerate(names_cov):
    names_cov[i]='cov'+str(i)

#===== STATISTICAL ANALYSIS: prepare dataframe to store statistical analysis results
stat=pd.DataFrame(data=None,
    columns=['LFC']+names_mean+names_cov+['SAMPLES'],
    index=np.arange(nlfc))
stat['LFC']=range(1,nlfc+1)


#===== STATISTICAL ANALYSIS
for i in range(1,nlfc+1):
    temp=ww[ww.LFC==i].drop('LFC',1)
    stat.ix[(stat.LFC==i),'SAMPLES']=temp.count()[0]
    stat.ix[stat.LFC==i,names_mean[0]:names_mean[-1]]=np.mean(temp.values,0)
    stat.ix[stat.LFC==i,names_cov[0]:names_cov[-1]]=np.cov(temp,rowvar=0).flatten()
    print ("LFC=%d, number of samples=%d" % (i, temp.count()[0]))
    pd.scatter_matrix(temp, color='k')
    plt.suptitle('LFC=%d' % i)
    print (temp.describe().ix['mean':'std'])

#===== MULTIVARIATE RANDOM NORMAL DISTRIBUTIONS
NN=300

mc=pd.DataFrame(data=None,
    columns=lognames0,
    index=np.arange(nlfc*NN), dtype='float')

for i in range(1,nlfc+1):
    mc.loc[NN*i-NN:NN*i-1,'LFC']=i
for i in range(1,nlfc+1):
    mean =             stat.loc[i-1,names_mean[0]:names_mean[-1]].values
    sigma = np.reshape(stat.loc[i-1,names_cov[0]:names_cov[-1]].values,(nlogs,nlogs))
    mc.ix[mc.LFC==i,1:] = np.random.multivariate_normal(mean,sigma,NN)


f, ax = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(12, 6))
scatt1=ax[0].scatter(ww.IP,ww.VPVS,20,ww.LFC,marker='o',edgecolors='none', alpha=0.2, cmap=cmap_facies,vmin=0,vmax=4)
scatt2=ax[1].scatter(mc.IP,mc.VPVS,20,mc.LFC,marker='o',edgecolors='none', alpha=0.5, cmap=cmap_facies,vmin=0,vmax=4)
ax[0].set_xlim(3000,9000); ax[0].set_ylim(1.5,3.0);
ax[0].set_title('augmented well data');
ax[1].set_title('synthetic data');
for i in range(len(ax)): ax[i].grid()   
uu=f.add_axes([0.95, 0.2, 0.015, 0.6])
cbar=f.colorbar(scatt2, cax=uu, ticks=[range(0,4+1)]);
cbar.set_alpha(1)
cbar.draw_all()

ww.to_csv('qsiwell2_augmented.csv',index=False)  # temporary output
mc.to_csv('qsiwell2_synthetic.csv',index=False)  # temporary output

x=mc.IP.values
y=mc.VPVS.values
z=mc.LFC.values

bb=7
plt.figure()
for i in range(1,nlfc+1):
    H, xedges, yedges = np.histogram2d(x[z==i], y[z==i], bins=bb)
    xi = np.linspace(np.min(x[z==i]), np.max(x[z==i]), H.shape[0])
    yi = np.linspace(np.min(y[z==i]), np.max(y[z==i]), H.shape[0])
    CS=plt.contour(xi, yi, H, 3, linewidths=1, colors=colors.rgb2hex(ccc[int(i)-1]), label=('LFC=%d' %i))
    plt.clabel(CS, inline=0, fontsize=10)
    # plt.contour(xi, yi, H, 2, linewidths=2, label=('LFC=%d' %i))

labels=[]
for i in range(1,nlfc+1):
    labels.append('LFC '+str(i))


for i in range(len(labels)):
    CS.collections[i].set_label(labels[i])
plt.legend(loc='upper left')

plt.legend(loc='best', fontsize=10)
plt.subplots_adjust(dimfig[0],dimfig[1],dimfig[2],dimfig[3])

#.......................... qui di seguito un accrocchio per discretizzare la colormap
colori = plt.get_cmap(colormap, nlfc)
cNorm=colors.Normalize(vmin=1, vmax=nlfc+1)
scalarMap=cmx.ScalarMappable(norm=cNorm,cmap=colori)
ccc=scalarMap.to_rgba(range(1,nlfc+1))
#......................................................................................


#********************************************************************
#********************************************************************
# codice matlab per controllare il fluid replacement

# aa=importdata('qsiwell2.csv',',',1);
# x=find(aa.data==-999.25);aa.data(x)=NaN;
# z=aa.data(:,2);
# gr=aa.data(:,5);
# vp=aa.data(:,3);
# vs=aa.data(:,4);
# rho=aa.data(:,8);
# sw=aa.data(:,9);
# vcl=aa.data(:,14);
# phi=aa.data(:,17);
# clear aa
# tmp=[z gr vp vs rho sw vcl phi];
# l=l_convert(tmp,{
#     'Depth','m','Measured Depth'
#     'gr','api','gamma ray'
#     'vp','m/s','Vp'
#     'vs','m/s','Vs'
#     'rho','g/cc','density'
#     'sw','v/v','water saturation'
#     'vcl','v/v','shale volume'
#     'phie','v/v','porosity'
#     });

# aux=l_plot(l,{'curves','vp','vs','rho','sw','phie','vcl'},{'color','k'});
# write_las_file(l,'INPUT/qsiwell2_matlab.las');

# # [kfl_i,rhofl_i,kvoigt_fl_i,vpb,rhofl_b,kfl_b,vpo,rhoo,ko,vpg,rhog,kg,gor]=flprop(0,80000,32,0.6,64,0,0,20,77,0.2,0.8);

# minerals=struct('name',{ 'clay', 'qtz', 'silt', 'calc'},...
#                 'k',   {  15,   37,  61.5,   76.8},...
#                 'mu',  {  5,   44.0,  41.1,   32.0},...
#                 'rho', {  2.81,  2.65,  2.79,   2.71})    ;
# respar={'p',20; 't',77; 'sal',80000; 'gg',0.6; 'og',32; 'gor',64; 'giib',0; 'giio',0};
# aafrm('qsiwell2_matlab.las',-1,9999,0.35,0.0,0,0,minerals,respar,'oil',1,1,0,3,.92,1);
# aafrm_qc('qsiwell2matlab_FRM.las',2100,2250,0.35,0.0,0)
