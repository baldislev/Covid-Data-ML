import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import copy as cp
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def prepare_data(data, training_data):

	#============================ AUXILIARY FUNCTIONS: =====================================

	# NEW FEATURES GENERATION:

	def extract_zip(addr):
		if len(addr.split(' ')) > 0:
			zipstr = addr.split(' ')[len(addr.split(' '))-1]
			if zipstr == 'nan':
				return 1
			return int(addr.split(' ')[len(addr.split(' '))-1])
		else:
			return int(addr)

	#date to number of days:
	def dateToDays(pcr_date):
		if pcr_date != 'nan':
			splitted = pcr_date.split('-')
			d0 = date(2020, 1, 1)
			d1 = date(int(splitted[0]), int(splitted[1]), int(splitted[2]))
			delta = d1 - d0
			return delta.days
		else:
			return np.nan

	def extract_x(loc):
		if len(loc.split(', ')) > 1:
			x = loc.strip('()').split(', ')[0].strip('\'\'')
			return np.float64(x)
		else:
			return np.nan

	def extract_y(loc):
		if len(loc.split(', ')) > 1:
			x = loc.strip('()').split(', ')[1].strip('\'\'')
			return np.float64(x)
		else:
			return np.nan

	def genCatConst(data, col, min, max, step):
		col_groups = np.arange(min, max, step)
		data[col+"_g"] = pd.cut(data[col], col_groups, include_lowest = True)

	def genCat(data, col, step):
		min = int(math.floor(data[col].min()))
		max = int(math.ceil (data[col].max()))
		col_groups = np.arange(min, max + step + 1, step) #arange is weird
		data[col+"_g"] = pd.cut(data[col], col_groups, include_lowest = True)

	def genGroups(data):
		genCat(data,'weight', 5)
		genCat(data,'household_income', 75)
		genCat(data,'sugar_levels', 5)
		genCat(data,'days', 20)
		genCatConst(data,'zip', 501, 101502, 1000)
		genCat(data, 'PCR_01', 0.1)
		genCat(data, 'PCR_02', 0.1)
		genCat(data, 'PCR_03', 20)
		genCat(data, 'PCR_04', 20)
		genCat(data, 'PCR_06', 1)
		genCat(data, 'PCR_07', 1.5)
		genCat(data, 'PCR_08', 0.5)
		genCat(data, 'PCR_09', 0.5)

	# OUTLIERS TREATMENT:

	def q25(x):
		return x.quantile(0.25)

	def q75(x):
		return x.quantile(0.75)

	def getUniFences(train_data, col):
		q1 = train_data[col].quantile(0.25)
		q3 = train_data[col].quantile(0.75)
		iqr = q3 - q1
		upper = q3 + 1.5*iqr
		lower = q1 - 1.5*iqr
		return upper, lower

	def cleanIQR(train_data, clean_d,col, toClean, to_round = False):
		q1 = train_data.groupby([col]).agg({toClean:[q25]})
		q3 = train_data.groupby([col]).agg({toClean:[q75]})
		q1 = q1[toClean]['q25']
		q3 = q3[toClean]['q75']
		iqr = q3 - q1
		upper = q3 + 1.5*iqr
		lower = q1 - 1.5*iqr
		uni_upper, uni_lower = getUniFences(train_data, toClean)
		lower = lower.where(lower > 0, 0)
		for i, row in clean_d.iterrows():
			colVal = clean_d.at[i, col]
			toCleanVal = clean_d.at[i, toClean]
			
			if (type(colVal) == pd._libs.interval.Interval) or (not math.isnan(colVal) and not math.isnan(toCleanVal)):
				if colVal in upper:  
					toCleanUpper = upper[colVal]
					toCleanLower = lower[colVal]
				else: 
					toCleanUpper, toCleanLower = uni_upper, uni_lower
				if toCleanVal > toCleanUpper:
					clean_d.at[i,toClean] = math.floor(toCleanUpper) if to_round else toCleanUpper
				elif toCleanVal < toCleanLower:
					clean_d.at[i,toClean] = math.ceil(toCleanLower) if to_round else toCleanLower

	def singleCleanIQR(train_data, clean_d, to_clean, to_round = False):
		q1 = train_data[to_clean].quantile(0.25)
		q3 = train_data[to_clean].quantile(0.75)
		iqr = q3 - q1
		upper, lower = getUniFences(train_data, to_clean)
		if to_round:
			upper = round(upper)
			lower = round(lower)

		for i, row in clean_d.iterrows():
			toCleanVal = clean_d.at[i, to_clean]
			if toCleanVal.astype(str) != 'nan':
				if toCleanVal > upper:
					clean_d.at[i,to_clean] = upper
				elif toCleanVal < lower:
					clean_d.at[i,to_clean] = lower


	def nanPercent(train_data, clean_d, to_clean, quant1, quant2, to_round = False):
		lower = train_data[to_clean].quantile(quant1)
		upper = train_data[to_clean].quantile(quant2)
		if to_round:
			upper = round(upper)
			lower = round(lower)
		for i, row in clean_d.iterrows():
			toCleanVal = clean_d.at[i, to_clean]
			if toCleanVal.astype(str) != 'nan':
				if toCleanVal > upper or toCleanVal < lower:
					clean_d.at[i,to_clean] = 'nan'


	def nanPCRs(clean_d, bad_days):
		bad_days_data = clean_d[clean_d.days_g == bad_days]
		for i, row in bad_days_data.iterrows():
			for pcr in pcr_list:
				clean_d.at[i, pcr] = 'nan'
		genGroups(clean_d)

	# DATA IMPUTATION

	def fillCoord(coord):
		mean = train_data[coord].mean()
		sig = train_data[coord].std()
		no_coord = clean_d[clean_d[coord].isna()]
		for i, row in no_coord.iterrows():
			clean_d.at[i, coord] = np.random.normal(mean, sig)

	def fillSexTrain():
		# no sex => fill by age and weight median
		# no age => only by weight median
		# no weight =>  only by age median
		# no age no weight, or alone in group => flip a coin
		medians1 = round(train_data.groupby(['age', 'weight_g']).agg({"F":[np.median]})['F']['median'])
		medians2 = round(train_data.groupby(['weight_g']).agg({"F":[np.median]})['F']['median'])
		medians3 = round(train_data.groupby(['age']).agg({"F":[np.median]})['F']['median'])
		no_sex = train_data[train_data.F.isna()]
		for i, row in no_sex.iterrows():
			age_val, weight_g_val = train_data.at[i, 'age'], train_data.at[i, 'weight_g']
			med1, med2, med3 = 'nan', 'nan', 'nan'
			F_val = np.nan
			if (age_val.astype(str) != 'nan'):
				if (weight_g_val.astype(str) != 'nan'):
					F_val = medians1[age_val][weight_g_val]
				else: # age, no weight
					F_val = medians3[age_val]
			else: # no age
				if (weight_g_val.astype(str) != 'nan'):
					F_val = medians2[weight_g_val]
			if F_val == np.nan:
				F_val = np.random.randint(0, high = 2 , size=1, dtype=int)
			train_data.at[i, 'F'] = F_val
	
	def randomVal(train_data, col, to_round = False):
		min, max = train_data[col].min(), train_data[col].max()
		rnd = np.random.uniform(min, high = max , size=1)[0]
		if to_round:
			rnd = round(rnd)
	  #if rnd < min or rnd > max:
	   # print ("BORDERLINE")
		return rnd

	def fillSexData():
		medians1 = round(train_data.groupby(['age', 'weight_g']).agg({"F":[np.median]})['F']['median'])
		no_sex = clean_d[clean_d.F.isna()]
		for i, row in no_sex.iterrows():
			age_val, weight_g_val = clean_d.at[i, 'age'], clean_d.at[i, 'weight_g']
			if (age, val) not in medians1:
				clean_d.at[i, 'F'] = randomVal(train_data, 'F', True)
				continue  
			clean_d.at[i, 'F'] = medians1[age_val][weight_g_val]

	def fillAgeTrain():
		#no age with job => more than 16 yo => median of more than 16
		job_age_med = train_data[(train_data.age > 16)].age.median()
		mask1 = (train_data.age.isna())& (train_data.job.isna() == False)
		train_data.loc[mask1, 'age'] = train_data.loc[mask1, 'age'].fillna(job_age_med)

		#no age no job => fill by weight group median
		weight_g_meds = train_data.groupby(['weight_g']).agg({"age":[np.median]})['age']['median']
		no_age_no_job = train_data[(train_data.age.isna()) & (train_data.job.isna()) & (train_data.weight.isna() == False)]
		for i, row in no_age_no_job.iterrows():
			train_data.at[i, 'age'] = round(weight_g_meds[train_data.at[i, 'weight_g']])

		# no age no job no weight => very few people => dont care => median age
		mask = (train_data.age.isna() & train_data.job.isna() & train_data.weight.isna())
		train_data.loc[mask, 'age'] = train_data.loc[mask, 'age'].fillna(round(train_data.age.median()))
	def fillAgeData():
		#no age with job => more than 16 yo => median of more than 16
		job_age_med = train_data[(train_data.age > 16)].age.median()
		mask1 = (clean_d.age.isna())& (clean_d.job.isna() == False)
		clean_d.loc[mask1, 'age'] = clean_d.loc[mask1, 'age'].fillna(job_age_med)

		#no age no job => fill by weight group median
		weight_g_meds = train_data.groupby(['weight_g']).agg({"age":[np.median]})['age']['median']
		no_age_no_job = clean_d[(clean_d.age.isna()) & (clean_d.job.isna())]
		for i, row in no_age_no_job.iterrows():
			weight_g_val = clean_d.at[i, 'weight_g']
			if weight_g_val not in weight_g_meds:
				clean_d.at[i, 'age'] = randomVal(train_data, 'age', True)
				continue  
			clean_d.at[i, 'age'] = round(weight_g_meds[weight_g_val])

	def fillWeightTrain():
		# no weight => fill by sex and age median
		ageF_g_meds = train_data.groupby(['age', 'F']).agg({"weight":[np.median]})['weight']['median']
		no_weight = train_data[train_data.weight.isna()]
		for i, row in no_weight.iterrows():
			train_data.at[i, 'weight'] = ageF_g_meds[train_data.at[i, 'age']][train_data.at[i, 'F']]

		# no age sex median => by age alone
		no_weight = train_data[train_data.weight.isna()]
		age_meds = train_data.groupby(['age']).agg({"weight":[np.median]})['weight']['median']
		for i, row in no_weight.iterrows():
			train_data.at[i, 'weight'] = age_meds[train_data.at[i, 'age']]
		genGroups(train_data)

	def fillWeightData():
		#clean_d
		# no weight => fill by sex and age median
		ageF_g_meds = train_data.groupby(['age', 'F']).agg({"weight":[np.median]})['weight']['median']
		no_weight = clean_d[clean_d.weight.isna()]
		for i, row in no_weight.iterrows():
			age_val, F_val = clean_d.at[i, 'weight_g'], clean_d.at[i, 'F']
			if (age_val, F_val) not in ageF_g_meds:
				clean_d.at[i, 'weight'] = randomVal(train_data, 'weight')
				continue  
			clean_d.at[i, 'weight'] = ageF_g_meds[age_val][F_val]
		genGroups(clean_d)

	def modeVal(train_data, col):
	  return train_data[col].value_counts().idxmax()

	def fillBlood():
		blood_mode = train_data.groupby(['F', 'age']).agg({'blood_type':['value_counts']})['blood_type']['value_counts']
		no_blood = clean_d[clean_d.blood_type.isna()]
		for i, row in no_blood.iterrows():
			sex_val, age_val = clean_d.at[i, 'F'], clean_d.at[i, 'age']
			if (sex_val, age_val) not in blood_mode:
				clean_d.at[i, 'blood_type'] = modeVal(train_data, 'blood_type')
				continue
			clean_d.at[i, 'blood_type'] = blood_mode[sex_val][age_val].idxmax()

	def fillZip():
		# uniform 0501 -> 99999
		no_zip = clean_d[(clean_d.zip == 1)]
		for i, row in no_zip.iterrows():
			clean_d.at[i, 'zip'] = np.random.randint(501, high = 10000 , size=1, dtype=int)

	def fillDays():
		# uniform min days -> max days
		no_date = clean_d[(clean_d.pcr_date.isna())]
		for i, row in no_date.iterrows():
			clean_d.at[i, 'days'] = np.random.randint(train_data['days'].min(), high = train_data['days'].max(), size=1, dtype=int)
		# gen days_g
		genGroups(clean_d)

	def fillPCR(train_data, clean_d, pcr):
		# fill PCR by normal distribution of the whole dataset
		mean = train_data[pcr].mean()
		std =  train_data[pcr].std()/4
		quant = 0.999 # empirically reasonable
		top_percentile = train_data[pcr].quantile(quant)
		min_percentle  = train_data[pcr].quantile(1 - quant)
		for i, row in clean_d.iterrows():
			if clean_d.at[i, pcr].astype(str) == 'nan':
				rnd = np.random.normal(mean, std)
				while rnd >= top_percentile or rnd <= min_percentle:
					rnd = np.random.normal(mean, std)
				if pcr in pcr_positive and rnd < 0:
					rnd = 0
				if pcr in pcr_dis:  # PCR_05, PCR_10 are discrete
					rnd = round(rnd)
				clean_d.at[i, pcr] = rnd
	def fillPCR1ByPCR2(train_data, clean_d, pcr1, pcr2):
		# Fill PCR1 by medians of PCR2 groups with this PCR1 value
		pcr1_meds = train_data.groupby([pcr2]).agg({pcr1:[np.median]})[pcr1]['median']
		no_pcr1 = clean_d[clean_d[pcr1].astype(str) == 'nan']
		for i, row in no_pcr1.iterrows():
			pcr2_val = clean_d.at[i, pcr2]
			pcr1_val = 'nan'
			if str(pcr2_val) != 'nan' and pcr2_val in pcr1_meds and pcr1_meds[pcr2_val].astype(str) != 'nan' :
				pcr1_val = pcr1_meds[pcr2_val]
			if pcr1 in pcr_dis and pcr1_val != 'nan' : # for discrete PCRs
				pcr1_val = round(pcr1_val)
			clean_d.at[i, pcr1] = pcr1_val

	def fillNan(train_data, clean_d, col_to_fill, emergency_col, to_round = False):
		# fill by every sex, age and weight_g group. If alone in one of the subgroups, use emergency_col
		# from best to worst:
		medians1 = train_data.groupby(['F', 'age', 'weight_g']).agg({col_to_fill:[np.median]})[col_to_fill]['median']
		medians2 = train_data.groupby(['F', 'age']).agg({col_to_fill:[np.median]})[col_to_fill]['median']
		medians3 = train_data.groupby(['age', 'weight_g']).agg({col_to_fill:[np.median]})[col_to_fill]['median']
		medians4 = train_data.groupby(['F', 'weight_g']).agg({col_to_fill:[np.median]})[col_to_fill]['median']
		medians5 = train_data.groupby([emergency_col]).agg({col_to_fill:[np.median]})[col_to_fill]['median']

		for i, row in clean_d.iterrows():
			if clean_d.at[i, col_to_fill].astype(str) == 'nan':
				sex_val, age_val, weight_g_val, emer_val = \
					clean_d.at[i, 'F'], clean_d.at[i, 'age'], clean_d.at[i, 'weight_g'],  clean_d.at[i, emergency_col]
				med_list = []
				for k in range(5):
					med_list.append(np.nan)
				if (sex_val, age_val, weight_g_val) in medians1:
					med_list[0] = medians1[sex_val][age_val][weight_g_val]
				if (sex_val, age_val) in medians2:
					med_list[1] = medians2[sex_val][age_val]
				if (age_val, weight_g_val) in medians3:
					med_list[2] = medians3[age_val][weight_g_val]
				if (sex_val, weight_g_val) in medians4:
					med_list[3] = medians4[sex_val][weight_g_val]
				if emer_val in medians5:
					med_list[4] = medians5[emer_val]
				#med_list = [med1, med2, med3, med4, med5]
				success = False
				for j in range(len(med_list)):
					if str(med_list[j]) != 'nan':
						clean_d.at[i, col_to_fill] = round(med_list[j]) if to_round else med_list[j]
						success = True
						break
				if not success:# EXTREMELY DRY train_data
					clean_d.at[i, col_to_fill] = randomVal(train_data, col_to_fill, to_round)


	
	# NORMALIZATION

	def normalize(train_data, clean_d):
		min_max_cols = ['weight','household_income', 'PCR_08', 'PCR_09', 'zip', 'x', 'y', 'conversations_per_day', 'sport_activity', 'days', 'num_of_siblings']
		z_score_cols = ['sugar_levels', 'PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_06', 'PCR_07', 'PCR_10']  

		mm_scaler = MinMaxScaler()
		mm_scaler.fit(train_data[min_max_cols])
		clean_d[min_max_cols] = mm_scaler.transform(clean_d[min_max_cols])

		z_scaler = StandardScaler()
		z_scaler.fit(train_data[z_score_cols])
		clean_d[z_score_cols] = z_scaler.transform(clean_d[z_score_cols])
	
	
	#================================ INITIALIZATION =========================================:

	clean_d = cp.deepcopy(data)
	train_data = cp.deepcopy(training_data)
	np.random.seed(5 + 9)
	
	pcr_list = ['PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_05', 'PCR_06', 'PCR_07', 'PCR_08', 'PCR_09', 'PCR_10']
	pcr_positive = ['PCR_03', 'PCR_04','PCR_05','PCR_10']
	pcr_dis = ['PCR_05', 'PCR_10']
	pcr_con = ['PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_06', 'PCR_07', 'PCR_08', 'PCR_09']


	#================================= NEW FEATURES GENERATION ======================================

	# Transform categorical features into OHE

	symptoms_ohe = clean_d['symptoms'].str.get_dummies(sep=";")
	clean_d.drop(labels=['symptoms'], axis=1, inplace=True)
	clean_d = clean_d.join(symptoms_ohe)

	sex_ohe = pd.get_dummies(clean_d['sex'])
	clean_d = clean_d.join(sex_ohe)

	sex_ohe = pd.get_dummies(train_data['sex'])
	train_data = train_data.join(sex_ohe)

	clean_d.drop(labels=['sex'], axis=1, inplace=True)

	# Extract new features from other features:
	for data in [train_data, clean_d]:
		data['zip']   = data['address'].astype(str).apply(lambda x : extract_zip(x))
		data['x'] = data['current_location'].astype(str).apply(lambda x : extract_x(x))
		data['y'] = data['current_location'].astype(str).apply(lambda x : extract_y(x))
		data['days'] = data['pcr_date'].astype(str).apply(lambda x : dateToDays(x))

		# Add groups for categorical data:
		genGroups(data)

	# ========================== OUTLIER TREATMENT ======================================
	
	# First of all, we don't trust the bad day's PCRs
	# During the bad days there are pcr values that we cannot rely on

	bad_days  = pd.Interval(140, 160, closed='right')
	good_days = pd.Interval(60, 80, closed='right')
	# Make all PCRs in bad days to have nan values
	nanPCRs(clean_d, bad_days)
	nanPCRs(train_data, bad_days)
	
	# Clean clean_d first, then train_data
	for data in [clean_d, train_data]:
		cleanIQR(train_data, data, 'age', 'weight')
		genGroups(data)
		cleanIQR(train_data, data, 'weight_g', 'age', True)
		cleanIQR(train_data, data, 'sport_activity', 'sugar_levels')

		#Goes to the pdf:
		cleanIQR(train_data, data, 'age', 'household_income')
		cleanIQR(train_data, data, 'conversations_per_day', 'happiness_score', True)# Happiness doesn't matter to conversations
		cleanIQR(train_data, data, 'happiness_score', 'conversations_per_day', True)
		cleanIQR(train_data, data, 'F', 'weight')
		genGroups(data)
		cleanIQR(train_data, data, 'weight_g', 'sugar_levels')
		genGroups(data)

		# PCRs:
		# now clean the PCRs:
		for pcr in ['PCR_01', 'PCR_02','PCR_03','PCR_04','PCR_05','PCR_06','PCR_07','PCR_10']:
			if pcr in pcr_dis: # discrete
				singleCleanIQR(train_data, data, pcr, True)
			else: singleCleanIQR(train_data, data, pcr)

		nanPercent(train_data, data, 'PCR_08', 0.0009, 0.9994)
		nanPercent(train_data, data, 'PCR_09', 0.0   , 0.999)
		genGroups(data)

	#========================================== DATA IMPUTATION: ===============================
	#Histo(clean_d)
	#Corr(clean_d, 'PCR_05', 'PCR_06')
	# Sex, age and weight are reliable group selection for other features
	# Therefore impute them first
	
	fillSexData()
	fillAgeData()
	fillWeightData()
	
	fillSexTrain()
	fillAgeTrain()
	fillWeightTrain()

	

	fillBlood()
	# Fill OHE blood features:
	blood_type_ohe = pd.get_dummies(clean_d['blood_type'])
	clean_d = clean_d.join(blood_type_ohe)


	fillZip()
	fillDays()
	fillCoord('x')
	fillCoord('y')

	# Impute train_data, then clean_d
	for data in [clean_d, train_data]:
		# impute by median of sex, age and weight groups

		fill_by_med_to_round = [
			'num_of_siblings', 'happiness_score',
			'conversations_per_day', 'sport_activity']
		fill_by_med = ['household_income', 'sugar_levels']

		for col_to_fill in fill_by_med:
			fillNan(train_data, data, col_to_fill, 'F')

		for col_to_fill in fill_by_med_to_round:
			fillNan(train_data, data, col_to_fill, 'F', True)

		genGroups(data)

		# Impute PCRS:
		# Impute dependent PCRs by one another:
		
		fillPCR1ByPCR2(train_data, data, 'PCR_01', 'PCR_02_g')
		genGroups(data)
		genGroups(train_data)
		
		fillPCR1ByPCR2(train_data, data, 'PCR_02', 'PCR_01_g')
		genGroups(data)
		genGroups(train_data)

		fillPCR1ByPCR2(train_data, data, 'PCR_05', 'PCR_06_g')
		genGroups(data)
		genGroups(train_data)

		fillPCR1ByPCR2(train_data, data, 'PCR_06', 'PCR_05')
		genGroups(data)
		genGroups(train_data)

		fillPCR1ByPCR2(train_data, data, 'PCR_03', 'PCR_04_g')
		genGroups(data)
		genGroups(train_data)
		fillPCR1ByPCR2(train_data, data, 'PCR_04', 'PCR_03_g')
		genGroups(data)
		genGroups(train_data)

	# fill the rest by normal distribution of the PCR:
	for pcr in pcr_list:
		fillPCR(train_data, clean_d, pcr)
		genGroups(data)
		genGroups(train_data)

	drop_list = ['patient_id', 'age', 'blood_type', 'address', 'current_location', 'job', 'pcr_date', 'M',
				 'happiness_score', 'PCR_05', 'AB+', 'AB-', 'B+', 'B-',
				 'weight_g', 'household_income_g', 'sugar_levels_g', 'days_g', 'zip_g',
				 'PCR_01_g', 'PCR_02_g', 'PCR_03_g', 'PCR_04_g', 'PCR_06_g', 'PCR_07_g',  'PCR_08_g',  'PCR_09_g', ]
	clean_d.drop(labels= drop_list, axis=1, inplace=True)
	
	# Normalization:
	normalize(train_data, clean_d)

	return clean_d



