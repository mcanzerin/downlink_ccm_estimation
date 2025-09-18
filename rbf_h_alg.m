function [J_COST,nmse_rbf_h_all,cmd_rbf_h_all,deviate_rbf_h_all]=rbf_h_alg(R_UL,R_DL,R_UL_train,R_DL_train,M_ant_num,num_train,num_test,num_real,max_iter,mu_1,mu_2,mu_reg)

J_COST=zeros(num_real,max_iter);

K_graph=5; %K nearest neighbors of a point will be connected to it, besides itself.

% Current_realization=0;
nmse_rbf_h_all=zeros(num_real,num_test);
cmd_rbf_h_all=zeros(num_real,num_test);
deviate_rbf_h_all=zeros(num_real,num_test);

% C_hat_RBF_h=zeros(M_ant_num,M_ant_num,num_test,num_real);

for i_real=1:num_real

        X_h_train=squeeze(R_UL_train(1,:,:,i_real));
        re_h_tr=real(X_h_train);
        im_h_tr=imag(X_h_train);
        X_h_train=[re_h_tr; im_h_tr(2:end,:)];

        X_h_test=squeeze(R_UL(1,:,:,i_real));
        re_h_ts=real(X_h_test);
        im_h_ts=imag(X_h_test);
        X_h_test=[re_h_ts; im_h_ts(2:end,:)];

        R_DL_train_vec=squeeze(R_DL_train(1,:,:,i_real));
        re_dl_tr=real(R_DL_train_vec);
        im_dl_tr=imag(R_DL_train_vec);
        R_DL_train1=[re_dl_tr; im_dl_tr(2:end,:)];
       
        % R_DL_test=squeeze(R_DL(1,:,:,i_real));
        % re_dl_ts=real(R_DL_test);
        % im_dl_ts=imag(R_DL_test);
        % R_DL_test1=[re_dl_ts; im_dl_ts(2:end,:)];
  
d_h=0;
X_h_diff=-1*ones(num_train);
X_h_diff_original=zeros(num_train);
for i=1:num_train
    a0=X_h_train(:,i)-X_h_train;
    b0=vecnorm(a0);
    c0=b0.^2;
    [c0_sorted,ind_c0_sorted]=sort(c0); 
    d_h=d_h+sqrt(c0_sorted(2));
    X_h_diff(i,ind_c0_sorted(2:K_graph+1))=c0_sorted(2:K_graph+1);
    X_h_diff(ind_c0_sorted(2:K_graph+1),i)=c0_sorted(2:K_graph+1);
    X_h_diff_original(i,ind_c0_sorted)=c0_sorted;
    X_h_diff_original(ind_c0_sorted,i)=c0_sorted;
end
d_h=d_h/num_train;

for i=1:num_train
    for j=1:num_train
        if X_h_diff(i,j)==-1
            X_h_diff(i,j)=Inf;
        end
    end
end

 
teta_h_prm_train=2.5;
Teta_h=teta_h_prm_train*d_h; 

W_w_h=exp(-X_h_diff./(Teta_h^2));

D_w_h=zeros(num_train);

for i=1:num_train

    D_w_h(i,i)=sum(W_w_h(i,:));
end

L_w_h=D_w_h-W_w_h;


%Initialization
%L_w_tilda is found above.

[s_h,Fi_h]=return_s_h_Fi_h(d_h,X_h_diff_original,num_train);

J_cost_h=zeros(length(s_h),1);

sigma_h_min_2=s_h(ceil(length(s_h)/2)); %Initial sigma_h^-2 value

X_h_test_diff=zeros(num_train,num_test); 
for i=1:num_test
    a02=X_h_test(:,i)-X_h_train;
    b02=vecnorm(a02);
    c02=b02.^2;
    X_h_test_diff(:,i)=c02;
end

Fi_h_0=exp(-X_h_diff_original.*sigma_h_min_2);

R_true=transpose(R_DL_train1);
% R_true_test=transpose(R_DL_test1);

A_mtx_H=L_w_h+mu_1*Fi_h_0^(-2); %FULL RANK

C_DIFFERENCE_h_NMSE=zeros(1,num_test);
C_DIFFERENCE_h_CMD=zeros(1,num_test);
C_DIFFERENCE_h_DEVIATE=zeros(1,num_test);

fh_inv2=zeros(size(Fi_h));

for i_h2=1:length(s_h) 
    fh_inv2(:,:,i_h2)=(Fi_h(:,:,i_h2))^(-2);
end

for iter_num=1:max_iter

%Optimization of Y_h
% THE CLOSED FORM SOLUTION!!!!
Y_h=mu_reg*(inv(A_mtx_H+mu_reg*eye(size(A_mtx_H))))*R_true;  %OPTIMIZATION OF Y_h

%Optimization of sigma_h  
    for i_h=1:length(s_h) 
        sigma_h_min_2=s_h(i_h);
        J_cost_h(i_h)=mu_1*(trace((transpose(Y_h))*(fh_inv2(:,:,i_h))*Y_h))+mu_2*sigma_h_min_2;
    end
[min_h,ind_h]=min(J_cost_h);
sigma_h_min_2=s_h(ind_h); %OPTIMIZATION OF sigma_h

Fi_h_0=Fi_h(:,:,ind_h); 

J_COST(i_real,iter_num)=trace((transpose(Y_h))*L_w_h*Y_h)+mu_1*trace((transpose(Y_h))*Fi_h_0^(-2)*Y_h)+mu_2*(sigma_h_min_2)+mu_reg*((norm((R_true-Y_h),'fro'))^2);
A_mtx_H=L_w_h+mu_1*Fi_h_0^(-2); %FULL RANK

end

C_h=(inv(Fi_h_0))*Y_h; %Kernel scale coefficients
Fi_h_test=exp(-X_h_test_diff.*sigma_h_min_2);
Y_h_test=transpose(Fi_h_test)*C_h;

for i=1:num_test
    if abs(Y_h_test(i,1))~=0
        Y_h_test(i,:)=Y_h_test(i,:)./abs(Y_h_test(i,1));
    end
end

re_yh_test=Y_h_test(:,1:M_ant_num);
im_yh_test=[zeros(num_test,1) Y_h_test(:,M_ant_num+1:end)];

R_DL_hat_h_test=re_yh_test+1i*im_yh_test;

%First Technique-Only h

for i=1:num_test
    c_test_1h=toeplitz(R_DL_hat_h_test(i,:));
    % c_test_0h=toeplitz(R_DL_hat_h_test(i,:));
    % c_test_1h=toeplitzify_fun(c_test_0h); 
    % C_hat_RBF_h(:,:,i,i_real)=c_test_1h;
    r_test_1h=R_DL(:,:,i,i_real);

    [V_DL_true,D_DL_true]=eig(r_test_1h);
    [d_DL_true,ind_DL_true]=max(diag(real(D_DL_true)));
    % w_DL_true=V_DL_true(:,ind_DL_true);

    [V_DL_RBF_h,D_DL_RBF_h]=eig(c_test_1h);
    [d_DL_RBF_h,ind_DL_RBF_h]=max(diag(real(D_DL_RBF_h)));
    w_DL_RBF_h=V_DL_RBF_h(:,ind_DL_RBF_h);

    C_DIFFERENCE_h_CMD(i)=1-((trace(c_test_1h*r_test_1h))/(norm(c_test_1h,'fro')*norm(r_test_1h,'fro')));
    C_DIFFERENCE_h_NMSE(i)=(norm((c_test_1h-r_test_1h),'fro')^2)/(norm(r_test_1h,'fro')^2);
    C_DIFFERENCE_h_DEVIATE(i)=1-((w_DL_RBF_h'*r_test_1h*w_DL_RBF_h)/d_DL_true);
end

nmse_rbf_h_all(i_real,:)=C_DIFFERENCE_h_NMSE;
cmd_rbf_h_all(i_real,:)=C_DIFFERENCE_h_CMD;
deviate_rbf_h_all(i_real,:)=C_DIFFERENCE_h_DEVIATE;

end

end