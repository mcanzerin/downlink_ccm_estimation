function [s_h,Fi_h]=return_s_h_Fi_h(d_h,X_h_diff_original,num_train)
    
    d_h_m2=d_h^-2;
    s_h0=logspace(log10(d_h_m2/100),log10(10*d_h_m2),300);
    end_s_h0_init=10;
    end_s_h0_const=end_s_h0_init;
    scale_up_s_h0=sqrt(10);
    s_h=[];

    while isempty(s_h)

        if end_s_h0_const~=end_s_h0_init
            % end_s_h0_const=end_s_h0_const*scale_up_s_h0;
            s_h0=logspace(log10(end_s_h0_init*d_h_m2),log10(end_s_h0_const*d_h_m2),50);
            end_s_h0_init=end_s_h0_const;
        end

        Fi_h0=exp(-kron(s_h0,X_h_diff_original));
        Fi_h0=reshape(Fi_h0,num_train,num_train,length(s_h0)); %CHECK!!!
    
        zero_h_cond=zeros(1,length(s_h0));
        zero_h_cond2=zeros(1,length(s_h0));

        for i_cond_h=1:length(s_h0)
            fffhhh=Fi_h0(:,:,i_cond_h);
            zero_h_cond(i_cond_h)=double(rcond(fffhhh)<eps);
            zero_h_cond2(i_cond_h)=double(rank(fffhhh)<num_train);
        end

        ind_fff_hhh=find(zero_h_cond==0&zero_h_cond2==0);
        Fi_h=Fi_h0(:,:,ind_fff_hhh);
        s_h=s_h0(ind_fff_hhh);
        end_s_h0_const=end_s_h0_const*scale_up_s_h0;

    end
end