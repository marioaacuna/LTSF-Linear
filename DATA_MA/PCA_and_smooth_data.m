% load data
convert_mat_to_csv('xNormF.mat', 'xNormF.csv');
%%
t =readtable('xNormF.csv');
% Do pca
[coeff,score,latent,tsquared,explained,mu] = pca(t_f, 'NumComponents', 1);
pca_data = score;
% Smooth data
smoothed_data = smooth(pca_data, 10);


% save smoothed data as mat file
save('xNormF_PCA_s.mat', 'smoothed_data');
convert_mat_to_csv('xNormF_PCA_s.mat', 'xNormF_PCA_s.csv');

%%