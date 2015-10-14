% %%SVM training inputs and outputs from libsvm 3.18 package
% %3.1 & 3.2 Function generator and code for training matrix and training labels
% Modify the value of the sigma and N samples 
%clc;
% c=[1,1;2,1.5;2,1;3,1.5];
% n=10;
% X=[];
% N = n;
% sigma=0.9;
% for i=1:4
% X=[X;sigma*randn(n,2)+repmat(c(i,:),n,1)];
% end
% Y=[ones(1,2*N) -ones(1,2*N)];
% plot(X(1:end/2,1),X(1:end/2,2),'+')
% hold all;
% plot(X(end/2+1:end,1),X(end/2+1:end,2),'o')
% hold off
% title('Sample data generated with Gaussians around different centroids'); 
% legend('label1', 'label2');
% 
% %%
% %3Construction of the classifier with the model parameters
% model = svmtrain(Y',X, '-s 0 -t 0 -c 100');
% %%Graphical representation of SVM
% [predicted_label, accuracy, prob_estimates] = svmpredict(Y',X,model);
% %%Dual Parameters w and b(rho value)
% w = model.SVs' * model.sv_coef;
% b = -model.rho;
% %Separating the data labels from the hyperplane %
% 
% if model.Label(1) == -1
%   w = -w;
%   b = -b;
% end
% 
% hold on 
% %Plotting the total number of Support vectors%
% sv = full(model.SVs);
% plot(sv(:,1),sv(:,2),'ko', 'MarkerSize', 10);
% %Finding the equation of the Classifer with Hyperplanes and margins...
% ...estimated from the sample data%
% y_hat = sign(w'*X' + b);
% plot_x = linspace(min(X(:,1)), max(X(:,1)), 30);
% plot_y = (-1/w(2))*(w(1)*plot_x + b);
% %Plotting the Margin lines for both labels%
% d_plus = plot_y + 1/norm(w); 
% d_minus = plot_y - 1/norm(w);
% %Margin separating the data with label +%
% plot(plot_x, d_plus, 'r--'); 
% hold on; 
% %Margin separating the data with label -%
% plot(plot_x, d_minus, 'b--'); 
% plot(plot_x, plot_y, 'k-', 'LineWidth', 1)
% title('SVM with hyperplane, margins and support vectors  with sigma=0.9'); 
% legend('+1', '-1', 'Support Vectors', 'margin', '-margin', 'Hyperplane', 'Location', 'Northwest')
%%
%3.3 Estimating the Structural Risk, Actual risk, Empirical risk for 
%%Call the DataGenerator function here to get N =10, 500 and sigma = 0.3
clc;
N = 10:1:500;
c = logspace(-1.5,1,100);
sigma = 0.3;
parameters ='-s 0 -t 0';

for k=1:500
     for l = 1:length(N)
%Splitting the 10 Dimensional data by random sampling into Training and Testing Data%
[x_train,y_train] = DataGenerator(N(l),sigma);
[x_test,y_test] = DataGenerator(N(l),sigma);
%Training the SVM%
model = svmtrain(y_train,x_train,parameters); 
%Calculation of Dual parameters%
w = model.SVs' * model.sv_coef;
b = -model.rho;

if model.Label(1) == -1
  w = -w;
  b = -b;
end

%Estimated labels%
y_hat = sign(x_train * w + b);
% %Calculation of training error or Empirical risk
emp_risk(l,k) = 0.5 / N(l) * sum(abs(y_hat - y_train));
%%
%Predicing the labels from the sample test data with SVMpredict%
[pred_label, accuracy,prob_est] = svmpredict(y_test,x_test,model);
%Calculation of testing error or Actual risk %
act_risk(l,k) = 0.5 / N(l) * sum(abs(pred_label - y_test));

    end
end

%Average of Training and Testing errors%
for p = 1 : length(N)
emp_risk_m(p) = mean(emp_risk(p,:));
act_risk_m(p) = mean(act_risk(p,:));
end
%%
%Plotting the Actual risk, Empirical risk,structural risk%
str_risk = emp_risk_m - act_risk_m;
figure;
plot(N,emp_risk_m, 'b', N, str_risk, 'r',N,act_risk_m,'g');
xlabel('No of Data Points');
ylabel('Calculated Risk');
title('Plots show the Empirical risk, Actual Risk and Structural risk');
legend('Empirical risk','Structural risk','Actual risk');
figure;
semilogx(N,emp_risk_m, 'b', N, str_risk, 'r',N,act_risk_m,'g');
xlabel('No of Data Points in Log scale');
ylabel('Calculated Risk in Log scale');
title('Plots show the Empirical risk, Actual Risk and Structural risk on a logarithmic scale m');
legend('Empirical risk','Structural risk','Actual risk');