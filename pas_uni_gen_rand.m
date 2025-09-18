function [R_UL,R_DL,R_UL_train,R_DL_train]=pas_uni_gen_rand(M_ant_num,num_test,num_train,num_real,D,alfa_ratio,teta,teta_train,delta_random,delta_random_train)

R_UL=zeros(M_ant_num,M_ant_num,num_test,num_real);
R_DL=zeros(M_ant_num,M_ant_num,num_test,num_real);

R_UL_train=zeros(M_ant_num,M_ant_num,num_train,num_real);
R_DL_train=zeros(M_ant_num,M_ant_num,num_train,num_real);


for i_real0=1:num_real

r_vec_UL=zeros(1,M_ant_num);
r_vec_DL=zeros(1,M_ant_num);

r_vec_UL_train=zeros(1,M_ant_num);
r_vec_DL_train=zeros(1,M_ant_num);


delta=delta_random;
delta_train=delta_random_train;


%Transmit correlation matrix R_mat is generated as follows: First,a row
%vector r_vec and a column vector c_vec are generated. Using them, a Toeplitz matrix is generated,
%which is R_mat. (R_mat depends on the difference between the row number and the column number.)
for k=1:num_test           
             

for cc1=1:M_ant_num
        fun1_UL=@(alfa) ((1/(2*delta(k,i_real0))).*exp(1i*2*pi*D*(cc1-1)*sin(alfa+teta(k,i_real0))));
        r_vec_UL(cc1)=integral(fun1_UL,-delta(k,i_real0),delta(k,i_real0));
        fun1_DL=@(alfa) ((1/(2*delta(k,i_real0))).*exp(1i*2*pi*alfa_ratio*D*(cc1-1)*sin(alfa+teta(k,i_real0))));        
        r_vec_DL(cc1)=integral(fun1_DL,-delta(k,i_real0),delta(k,i_real0));       
end


R_UL(:,:,k,i_real0)=toeplitz(r_vec_UL)./r_vec_UL(1);
R_DL(:,:,k,i_real0)=toeplitz(r_vec_DL)./r_vec_DL(1);


end



for k=1:num_train           


for cc1=1:M_ant_num
        fun1_UL=@(alfa) ((1/(2*delta_train(k,i_real0))).*exp(1i*2*pi*D*(cc1-1)*sin(alfa+teta_train(k,i_real0))));
        r_vec_UL_train(cc1)=integral(fun1_UL,-delta_train(k,i_real0),delta_train(k,i_real0));
        fun1_DL=@(alfa) ((1/(2*delta_train(k,i_real0))).*exp(1i*2*pi*alfa_ratio*D*(cc1-1)*sin(alfa+teta_train(k,i_real0))));        
        r_vec_DL_train(cc1)=integral(fun1_DL,-delta_train(k,i_real0),delta_train(k,i_real0));       
end


R_UL_train(:,:,k,i_real0)=toeplitz(r_vec_UL_train)./r_vec_UL_train(1);
R_DL_train(:,:,k,i_real0)=toeplitz(r_vec_DL_train)./r_vec_DL_train(1);


end


end


end

