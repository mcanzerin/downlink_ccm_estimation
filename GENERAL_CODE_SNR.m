clearvars;
% close all;
rng("shuffle")
% tic
%%
% PARAMETERS FOR EXPERIMENTS

% 1-Communication Parameters

M_ant_num=64;  %Number of Base Station (BS) Antennas
SNR_dB_vec=0:10:40; %SNR for Creating CCM Dataset

% 2-Algorithm Parameters

% % %20 dB-64 antennas
mu_norm=1e6;
mu_1=1e5/mu_norm;  
mu_2=3e11/mu_norm; 
mu_reg=1e8/mu_norm; 

K_user_num=100; 
num_train=4*K_user_num;
num_test=K_user_num;

max_iter=30;

%% CCM GENERATION PARAMETERS

num_prm=1;
num_real=5;

%System Configuration in the simulations
% M_ant_num=64;      
D=0.5;
f_dl=2.14; %DL Carrier Frequency (GHz)
f_ul=1.95; %UL Carrier Frequency (GHz)
alfa_ratio=(f_dl/f_ul); %DL/UL Carrier Frequency Ratio

% PAS Parameters
delta_min=5; %degree
delta_max=15; %degree
% delta_cons=5; %degree

teta_min=-pi;
teta_max=pi;

teta=teta_min+(teta_max-teta_min)*rand(K_user_num,num_real); %azimuth angle
teta_train=teta_min+(teta_max-teta_min)*rand(num_train,num_real); %azimuth angle 

delta_random=(delta_min+(delta_max-delta_min)*rand(K_user_num,num_real))*pi/180; %angle spread
delta_random_train=(delta_min+(delta_max-delta_min)*rand(num_train,num_real))*pi/180; %angle spread

%% PAS: Uniform

% %Random Angular Spread
[R_UL0,R_DL0,R_UL_train0,R_DL_train0]=pas_uni_gen_rand(M_ant_num,num_test,num_train,num_real,D,alfa_ratio,teta,teta_train,delta_random,delta_random_train);

save ("Dataset_for_SNR_No_pp_v8_5DS.mat","teta","teta_train","delta_random","delta_random_train")
%%
NMSE_RBF_h_mean_vec=zeros(1,length(SNR_dB_vec));
CMD_RBF_h_mean_vec=zeros(1,length(SNR_dB_vec));
DEVIATE_RBF_h_mean_vec=zeros(1,length(SNR_dB_vec));

NMSE_dict_mean_vec=zeros(1,length(SNR_dB_vec));
CMD_dict_mean_vec=zeros(1,length(SNR_dB_vec));
DEVIATE_dict_mean_vec=zeros(1,length(SNR_dB_vec));

NMSE_sinc_mean_vec=zeros(1,length(SNR_dB_vec));
CMD_sinc_mean_vec=zeros(1,length(SNR_dB_vec));
DEVIATE_sinc_mean_vec=zeros(1,length(SNR_dB_vec));


nmse_rbf_h_all_ALL_SNR=zeros(num_real,num_test,length(SNR_dB_vec));
cmd_rbf_h_all_ALL_SNR=zeros(num_real,num_test,length(SNR_dB_vec));
deviate_rbf_h_all_ALL_SNR=zeros(num_real,num_test,length(SNR_dB_vec));

nmse_dict_all_ALL_SNR=zeros(num_real,num_test,length(SNR_dB_vec));
cmd_dict_all_ALL_SNR=zeros(num_real,num_test,length(SNR_dB_vec));
deviate_dict_all_ALL_SNR=zeros(num_real,num_test,length(SNR_dB_vec));

Q_T_VEC_ALL_SNR=zeros(num_real,num_test,length(SNR_dB_vec));

nmse_sinc_all_ALL_SNR=zeros(num_real,num_test,length(SNR_dB_vec));
cmd_sinc_all_ALL_SNR=zeros(num_real,num_test,length(SNR_dB_vec));
deviate_sinc_all_ALL_SNR=zeros(num_real,num_test,length(SNR_dB_vec));

% parpool(2)
parfor i_SNR_dB=1:length(SNR_dB_vec)
% for i_SNR_dB=1:length(SNR_dB_vec)
    SNR_dB=SNR_dB_vec(i_SNR_dB); %SNR for Creating CCM Dataset

%Generate Imperfect CCM Dataset
R_UL_original=R_UL0;
R_DL_original=R_DL0;
R_UL_train_original=R_UL_train0;
R_DL_train_original=R_DL_train0;

[R_UL,R_UL_train,R_DL_train]=imperfect_dataset_gen(SNR_dB,M_ant_num,num_test,num_train,num_real,R_UL0,R_UL_train0,R_DL_train0);
R_DL=R_DL0;

%ALGORITHMS

%1-RBF-h:
[J_COST,nmse_rbf_h_all,cmd_rbf_h_all,deviate_rbf_h_all]=rbf_h_alg(R_UL,R_DL,R_UL_train,R_DL_train,M_ant_num,num_train,num_test,num_real,max_iter,mu_1,mu_2,mu_reg);

NMSE_RBF_h_mean_vec(i_SNR_dB)=mean(nmse_rbf_h_all,"all");
CMD_RBF_h_mean_vec(i_SNR_dB)=real(mean(cmd_rbf_h_all,"all"));
DEVIATE_RBF_h_mean_vec(i_SNR_dB)=real(mean(deviate_rbf_h_all,"all"));

nmse_rbf_h_all_ALL_SNR(:,:,i_SNR_dB)=nmse_rbf_h_all;
cmd_rbf_h_all_ALL_SNR(:,:,i_SNR_dB)=cmd_rbf_h_all;
deviate_rbf_h_all_ALL_SNR(:,:,i_SNR_dB)=deviate_rbf_h_all;

%2-Dictionary:
[nmse_dict_all,cmd_dict_all,deviate_dict_all,Q_T_VEC]=dict_euc_mir_alg(R_UL,R_DL,R_UL_train,R_DL_train,M_ant_num,num_train,num_test,num_real);


NMSE_dict_mean_vec(i_SNR_dB)=mean(nmse_dict_all,"all");
CMD_dict_mean_vec(i_SNR_dB)=real(mean(cmd_dict_all,"all"));
DEVIATE_dict_mean_vec(i_SNR_dB)=real(mean(deviate_dict_all,"all"));

Q_T_VEC_ALL_SNR(:,:,i_SNR_dB)=Q_T_VEC;
nmse_dict_all_ALL_SNR(:,:,i_SNR_dB)=nmse_dict_all;
cmd_dict_all_ALL_SNR(:,:,i_SNR_dB)=cmd_dict_all;
deviate_dict_all_ALL_SNR(:,:,i_SNR_dB)=deviate_dict_all;

%3-Sinc Transformation:
[nmse_sinc_all,cmd_sinc_all,deviate_sinc_all]=sinc_alg(R_UL,R_DL,alfa_ratio,M_ant_num,num_test,num_real);

NMSE_sinc_mean_vec(i_SNR_dB)=mean(nmse_sinc_all,"all");
CMD_sinc_mean_vec(i_SNR_dB)=real(mean(cmd_sinc_all,"all"));
DEVIATE_sinc_mean_vec(i_SNR_dB)=real(mean(deviate_sinc_all,"all"));

nmse_sinc_all_ALL_SNR(:,:,i_SNR_dB)=nmse_sinc_all;
cmd_sinc_all_ALL_SNR(:,:,i_SNR_dB)=cmd_sinc_all;
deviate_sinc_all_ALL_SNR(:,:,i_SNR_dB)=deviate_sinc_all;

end


save ("Results_for_SNR_No_pp_v8_5DS.mat","SNR_dB_vec","NMSE_RBF_h_mean_vec","NMSE_dict_mean_vec","NMSE_sinc_mean_vec","CMD_RBF_h_mean_vec","CMD_dict_mean_vec","CMD_sinc_mean_vec","DEVIATE_RBF_h_mean_vec","DEVIATE_dict_mean_vec","DEVIATE_sinc_mean_vec","nmse_rbf_h_all_ALL_SNR","cmd_rbf_h_all_ALL_SNR","deviate_rbf_h_all_ALL_SNR","nmse_dict_all_ALL_SNR","cmd_dict_all_ALL_SNR","deviate_dict_all_ALL_SNR","Q_T_VEC_ALL_SNR","nmse_sinc_all_ALL_SNR","cmd_sinc_all_ALL_SNR","deviate_sinc_all_ALL_SNR")

figure 
semilogy(SNR_dB_vec,NMSE_RBF_h_mean_vec,'-o')
hold on
semilogy(SNR_dB_vec,NMSE_dict_mean_vec,'-x')
semilogy(SNR_dB_vec,NMSE_sinc_mean_vec,'-square')
xlabel("SNR (dB)")
ylabel("Average NMSE")
legend('Proposed Algorithm','Dictionary','Sinc Transformation')

figure 
semilogy(SNR_dB_vec,CMD_RBF_h_mean_vec,'-o')
hold on
semilogy(SNR_dB_vec,CMD_dict_mean_vec,'-x')
semilogy(SNR_dB_vec,CMD_sinc_mean_vec,'-square')
xlabel("SNR (dB)")
ylabel("Average CMD")
legend('Proposed Algorithm','Dictionary','Sinc Transformation')

figure 
semilogy(SNR_dB_vec,DEVIATE_RBF_h_mean_vec,'-o')
hold on
semilogy(SNR_dB_vec,DEVIATE_dict_mean_vec,'-x')
semilogy(SNR_dB_vec,DEVIATE_sinc_mean_vec,'-square')
xlabel("SNR (dB)")
ylabel("Average DM")
legend('Proposed Algorithm','Dictionary','Sinc Transformation')
