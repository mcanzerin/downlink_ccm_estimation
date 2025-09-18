function [R_UL,R_UL_train,R_DL_train]=imperfect_dataset_gen(SNR_dB,M_ant_num,num_test,num_train,num_real,R_UL,R_UL_train,R_DL_train)

% Create random channels

SNR=10^(SNR_dB/10); % linear
sigma_noise=sqrt(M_ant_num/SNR); %TAKEN INTO ACCOUNT WHILE CALCULATING THE CHANNEL REALIZATIONS!!!!!
N_ch=2*M_ant_num; %Number of channel realizations

UCCM_sample_avg=zeros(M_ant_num,M_ant_num,num_test,num_real);

UCCM_sample_avg_proj=zeros(M_ant_num,M_ant_num,num_test,num_real);

UCCM_sample_avg_train=zeros(M_ant_num,M_ant_num,num_train,num_real);
DCCM_sample_avg_train=zeros(M_ant_num,M_ant_num,num_train,num_real);

UCCM_sample_avg_proj_train=zeros(M_ant_num,M_ant_num,num_train,num_real);
DCCM_sample_avg_proj_train=zeros(M_ant_num,M_ant_num,num_train,num_real);

for i_real=1:num_real

w_k_UL = (1/sqrt(2))*(randn(M_ant_num,N_ch,num_test)+1i*randn(M_ant_num,N_ch,num_test));

n_k_UL = (sqrt(M_ant_num/SNR)/sqrt(2))*(randn(M_ant_num,N_ch,num_test)+1i*randn(M_ant_num,N_ch,num_test));

w_k_UL_train = (1/sqrt(2))*(randn(M_ant_num,N_ch,num_train)+1i*randn(M_ant_num,N_ch,num_train));
w_k_DL_train = (1/sqrt(2))*(randn(M_ant_num,N_ch,num_train)+1i*randn(M_ant_num,N_ch,num_train));

n_k_UL_train = (sqrt(M_ant_num/SNR)/sqrt(2))*(randn(M_ant_num,N_ch,num_train)+1i*randn(M_ant_num,N_ch,num_train));
n_k_DL_train = (sqrt(M_ant_num/SNR)/sqrt(2))*(randn(M_ant_num,N_ch,num_train)+1i*randn(M_ant_num,N_ch,num_train));

for k=1:num_test
    h_k_UL=(sqrtm(R_UL(:,:,k,i_real)))*w_k_UL(:,:,k)+n_k_UL(:,:,k);
    UCCM_sample=zeros(M_ant_num,M_ant_num,N_ch);
    for nc=1:N_ch
        UCCM_sample(:,:,nc)=h_k_UL(:,nc)*h_k_UL(:,nc)';
    end
    UCCM_sample_avg(:,:,k,i_real)=mean(UCCM_sample,3)-(sigma_noise^2)*eye(M_ant_num);
end


% nmse_ul_imp=zeros(1,num_test);

for k=1:num_test
    A_uccm=UCCM_sample_avg(:,:,k,i_real);
    UCCM_sample_avg_proj(:,:,k,i_real)=toeplitzify_fun(A_uccm);
    % nmse_ul_imp(k)=(norm(R_UL(:,:,k,i_real)-UCCM_sample_avg_proj(:,:,k,i_real),'fro')/norm(R_UL(:,:,k,i_real),'fro'))^2;
end


for k=1:num_train
    h_k_UL_train=(sqrtm(R_UL_train(:,:,k,i_real)))*w_k_UL_train(:,:,k)+n_k_UL_train(:,:,k);
    h_k_DL_train=(sqrtm(R_DL_train(:,:,k,i_real)))*w_k_DL_train(:,:,k)+n_k_DL_train(:,:,k);
    UCCM_sample_train=zeros(M_ant_num,M_ant_num,N_ch);
    DCCM_sample_train=zeros(M_ant_num,M_ant_num,N_ch);
    for nc=1:N_ch
        UCCM_sample_train(:,:,nc)=h_k_UL_train(:,nc)*h_k_UL_train(:,nc)';
        DCCM_sample_train(:,:,nc)=h_k_DL_train(:,nc)*h_k_DL_train(:,nc)';
    end
    UCCM_sample_avg_train(:,:,k,i_real)=mean(UCCM_sample_train,3)-(sigma_noise^2)*eye(M_ant_num);
    DCCM_sample_avg_train(:,:,k,i_real)=mean(DCCM_sample_train,3)-(sigma_noise^2)*eye(M_ant_num);
end


for k=1:num_train
    A_uccm_train=UCCM_sample_avg_train(:,:,k,i_real);
    UCCM_sample_avg_proj_train(:,:,k,i_real)=toeplitzify_fun(A_uccm_train);
    A_dccm_train=DCCM_sample_avg_train(:,:,k,i_real);
    DCCM_sample_avg_proj_train(:,:,k,i_real)=toeplitzify_fun(A_dccm_train);
end


end


R_UL=UCCM_sample_avg_proj;

R_UL_train=UCCM_sample_avg_proj_train;
R_DL_train=DCCM_sample_avg_proj_train;


end
