#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:18:27 2023
@author: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
"""

import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm

from fair_tools import runFaIR, read_calib_samples, read_forcing_data

cwd = os.getcwd()

figdir=f"{cwd}/figures/natural_variability/"
datadir=f"{cwd}/data"

fair_calibration_dir=f"{cwd}/fair-calibrate"
fair_2_dir=f"{cwd}/leach-et-al-2021"

hadcrut5_datadir=f"{fair_2_dir}/data/input-data/Temperature-observations"



data=dict()
mean=dict()
std=dict()
# data['HadCrut5']=xr.open_dataset(f'{datadir}/HadCRUT.5.0.1.0.analysis.summary_series.global.annual.nc')
data['HadCrut5-m']=xr.open_dataset(f'{hadcrut5_datadir}/HadCRUT.5.0.1.0.analysis.ensemble_series.global.monthly.nc')
data['HadCrut5']=data['HadCrut5-m'].tas.groupby('time.year').mean('time')


# Change reference temperature to mean of 1851-1900
data['HadCrut5']=data['HadCrut5']-data['HadCrut5'].loc[dict(year=slice(1850,1900))].mean(dim='year')



def detrend_xarray(data_array, dim, order=4, return_trend=False):
    """
    Generated with ChatGPT4
    Detrends an xarray DataArray using a polynomial of specified order.

    :param data_array: xarray DataArray with time series data.
    :param dim: Name of the dimension along which to detrend (e.g., 'time').
    :param order: Order of the polynomial to be used for detrending. Default is 4.
    :return: xarray DataArray of detrended data.
    """
    if not isinstance(data_array, xr.DataArray):
        raise ValueError("Input must be an xarray DataArray.")

    # Extracting the dimension values (e.g., time)
    dim_values = data_array[dim].values

    # Reshaping for polynomial fitting
    X = np.array(dim_values).reshape(-1, 1)

    # Create polynomial features
    poly = PolynomialFeatures(degree=order)
    X_poly = poly.fit_transform(X)

    # Fit the model
    model = LinearRegression()
    model.fit(X_poly, data_array.values)

    # Predict the trend
    trend = model.predict(X_poly)

    # Detrend the data
    detrended = data_array - trend

    if return_trend:
        return trend, detrended
    else:
        return detrended

# Calculate std for each HadCrut5 realization
number_of_realizations=data['HadCrut5'].shape[0]
std_hadcrut5=np.zeros(number_of_realizations)
detrended_hadcrut=xr.zeros_like(data['HadCrut5'])
trend_hadcrut=xr.zeros_like(data['HadCrut5'])
for i in range(1,number_of_realizations+1):
    trend_hadcrut.loc[dict(realization=i)], detrended_hadcrut.loc[dict(realization=i)]=detrend_xarray(data['HadCrut5'].loc[dict(realization=i)],'year', return_trend=True)
    std_hadcrut5[i-1]=float(detrended_hadcrut.loc[dict(realization=i)].std())
    

# Load FaIR calibration and carry out model runs
calib_configs = read_calib_samples()
scenario = 'ssp245'
solar_forcing, volcanic_forcing, emissions = read_forcing_data(scenario,len(calib_configs),1750,2100)
fair_calib = runFaIR(solar_forcing,volcanic_forcing,emissions,calib_configs,scenario,
                     start=1750,end=2100)

#Make another ensemble without stochasticity
fair_calib_det = runFaIR(solar_forcing,volcanic_forcing,emissions,calib_configs,scenario,
                     start=1750,end=2100, stochastic_run=False)


#Calculate natural variability for FaIR runs
number_of_fair_runs=fair_calib.temperature.shape[2]
std_fair=np.zeros(number_of_fair_runs)
detrended_fair=xr.zeros_like(fair_calib.temperature.loc[dict(layer=0)])
trend_fair=xr.zeros_like(fair_calib.temperature.loc[dict(layer=0)])

std_fair_det=np.zeros(number_of_fair_runs)
detrended_fair_det=xr.zeros_like(fair_calib.temperature.loc[dict(layer=0)])
trend_fair_det=xr.zeros_like(fair_calib.temperature.loc[dict(layer=0)])


for i, config in enumerate(calib_configs.index):
    trend_fair.loc[dict(config=config)], detrended_fair.loc[dict(config=config)]=detrend_xarray(fair_calib.temperature.loc[dict(layer=0, config=config) ],'timebounds', return_trend=True)
    std_fair[i]=float((detrended_fair.loc[dict(config=config)].std()))
    
    trend_fair_det.loc[dict(config=config)], detrended_fair_det.loc[dict(config=config)]=detrend_xarray(fair_calib_det.temperature.loc[dict(layer=0, config=config) ],'timebounds', return_trend=True)
    std_fair_det[i]=float((detrended_fair_det.loc[dict(config=config)].std()))

# Calculate correlation coefficients for all parameters
corrcoeff=pd.Series(index=calib_configs.columns, dtype='float64')
corrcoeff_above_threshold=pd.Series(dtype='float64')
corrcoeff_threshold=0.1
for i, param in enumerate(corrcoeff.index):
    corrcoeff[param]=np.corrcoef(calib_configs[param], std_fair)[0,1]
    if abs(corrcoeff[param])>corrcoeff_threshold:
        corrcoeff_above_threshold[param]=corrcoeff[param]
        print(param+' = ' + str(round(corrcoeff[param],3)))

# Plot examples of detrending 
colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:brown'] 
fig0, ax0=pl.subplots(2,2, figsize=(15,10), sharex=True)
for i, color in zip([15], colors):
    data['HadCrut5'].loc[dict(realization=i)].plot(ax=ax0[0,0], color=color)
    trend_hadcrut.loc[dict(realization=i)].plot(ax=ax0[0,0])
    detrended_hadcrut.loc[dict(realization=i)].plot(ax=ax0[1,0])
    
    config=fair_calib.configs[i]
    fair_calib.temperature.loc[dict(layer=0, config=config)].plot(ax=ax0[0,1], color=color )
    # trend_fair.loc[dict(config=config)].plot(ax=ax0[0,1], color=color)
    detrended_fair.loc[dict(config=config)].plot(ax=ax0[1,1], color=color)
    
    fair_calib_det.temperature.loc[dict(layer=0, config=config)].plot(ax=ax0[0,1], color=colors[1] )
    # trend_fair_det.loc[dict(config=config)].plot(ax=ax0[0,1], color=colors[1])
    detrended_fair_det.loc[dict(config=config)].plot(ax=ax0[1,1], color=colors[1])
    
    ax0[0,0].set_title('Hadcrut5')
    ax0[0,0].set_xlim(1850,2025)
    ax0[0,1].set_ylim(-0.3,1.0)
    ax0[0,0].set_ylim(-0.3,1.0)
    ax0[1,1].set_ylim(-0.4,0.6)
    ax0[1,0].set_ylim(-0.4,0.6)
    
fig0.savefig(f'{figdir}/detrending_examples.png', dpi=150)    
    

# Plot histograms of internal variability in HadCrut5 and FaiR runs
fig1, ax1=pl.subplots(1,1)
ax1.hist(std_hadcrut5, bins=20, alpha=0.5, density=True, label='HadCrut5') 
ax1.hist(std_fair, bins=20, alpha=0.5, density=True, label='FaIR Calibration - stochastic')
ax1.hist(std_fair_det, bins=20, alpha=0.5, density=True, label='FaIR Calibration - deterministic')
ax1.legend()
ax1.set_xlabel('Internal variability (std)')
fig1.savefig(f'{figdir}/internal_variability_histograms.png', dpi=150)

# Make scatter plots with FaIR internal variability vs sigma_xi and sigma_eta
fig2, ax2=pl.subplots(1,4, sharey=True, figsize=(14,3))
ax2[0].scatter(calib_configs['sigma_xi'], std_fair, s=4)
ax2[0].set_title('sigma_xi \n (correlation = '+ str(round(np.corrcoef(calib_configs['sigma_xi'], std_fair)[0,1],3))+')')
ax2[0].set_xlabel('sigma_xi')
ax2[0].set_ylabel('Internal variability')

ax2[1].scatter(calib_configs['sigma_eta'], std_fair, s=4)
ax2[1].set_title('sigma_eta \n (correlation = '+ str(round(np.corrcoef(calib_configs['sigma_eta'], std_fair)[0,1],3))+')')
ax2[1].set_xlabel('sigma_eta')

ax2[2].scatter(calib_configs['scale Volcanic'], std_fair, s=4)
ax2[2].set_title('scale Volcanic \n (correlation = '+ str(round(np.corrcoef(calib_configs['scale Volcanic'], std_fair)[0,1],3))+')')
ax2[2].set_xlabel('scale Volcanic')

ax2[3].scatter(calib_configs['gamma'], std_fair, s=4)
ax2[3].set_title('gamma \n (correlation = '+ str(round(np.corrcoef(calib_configs['gamma'], std_fair)[0,1],3))+')')
ax2[3].set_xlabel('gamma')

fig2.savefig(f'{figdir}/scatter_internal_variability.png', dpi=150)


# Make scatter plots with FaIR internal variability vs all parameters
fig3, ax3=pl.subplots(3,5, sharey=True, figsize=(10,6))
ax3_flat=ax3.flatten()
for i, param in enumerate(corrcoeff_above_threshold.index):
    ax3_flat[i].scatter(calib_configs[param], std_fair, s=0.5)
    ax3_flat[i].set_title(param+'\n (correlation = '+ str(round(corrcoeff_above_threshold[param],3))+')')
    ax3_flat[i].set_xlabel(param)
    # ax3_flat[i].set_aspect('equal') #, adjustable='box')  # Set equal aspect ratio
    ax3_flat[i].set_aspect(1./ax3_flat[i].get_data_ratio())
    ax3_flat[i].set_ylabel('Internal variability')
ax3_flat[-1].axis('off')
fig3.tight_layout()
fig3.savefig(f'{figdir}/scatter_internal_variability_14vars.png', dpi=150)


# Fit a normal distribution to internal variability of HadCRUT5
mu_hadcrut5_variability, std_hadcrut5_variability =  norm.fit(std_hadcrut5)
norm_hadcrut5_variability=norm(loc=mu_hadcrut5_variability, scale=std_hadcrut5_variability)

#Calculate weights based on internal variability
weights=pd.Series(index=calib_configs.index, dtype='float64')
for i in range(len(weights)):
    weights.iloc[i]=norm_hadcrut5_variability.pdf(std_fair[i])
# normalize weights
weights=weights/weights.sum()

# Plot PDFs for parameters before and after applying weights
fig4, ax4=pl.subplots(7,7, figsize=(18,14))
ax4_flat=ax4.flatten()
for i, param in enumerate(calib_configs.columns):
    calib_configs[param].plot.hist(density=True, bins=25, alpha=0.7, ax=ax4_flat[i], label='FaIR calib 1.0.2')
    calib_configs[param].plot.hist(density=True, bins=25, alpha=0.7, ax=ax4_flat[i], weights=weights, label='FaIR calib 1.0.2 weighted with variability')
    ax4_flat[i].set_title(param)
    
# ax4_flat[-3].legend() # I could not get the legend showing nicely, orange is weighted
ax4_flat[-1].axis('off')
ax4_flat[-2].axis('off')
fig4.tight_layout()
fig4.savefig(f'{figdir}/parameter_pdfs_weighted_with_internal_variability.png', dpi=150)

#Plot distributions of year-2100 temperature with and without weighting 
fig5, ax5=pl.subplots(1,2)
fair_calib.temperature.loc[dict(layer=0,timebounds=2020)].plot.hist(bins=25,density=True, ax=ax5[0], label='Non-weighted', alpha=0.7)
fair_calib.temperature.loc[dict(layer=0,timebounds=2020)].plot.hist(bins=25,weights=weights, density=True, ax=ax5[0], label='Weighted', alpha=0.7)
ax5[0].set_title('Temperature at 2020')
ax5[0].legend()


fair_calib.temperature.loc[dict(layer=0,timebounds=2100)].plot.hist(bins=25,density=True, ax=ax5[1], label='Non-weighted', alpha=0.7)
fair_calib.temperature.loc[dict(layer=0,timebounds=2100)].plot.hist(bins=25,weights=weights, density=True, ax=ax5[1], label='Weighted', alpha=0.7)
ax5[1].set_title('Temperature at 2100')
ax5[1].legend()
fig5.savefig(f'{figdir}/temperature_distribution_2020_and_2100.png', dpi=150)
