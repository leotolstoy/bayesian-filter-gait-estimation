close all;clc;clearvars -except Gaitcycle Continuous %removes all variables except for gaitcyle and continuous from the matlab workspace. Useful because loading these variables can take several minutes.
%load the InclineExperiment data from the folder location you specify
% load('Z:\your_file_location_here\InclineExperiment.mat') 
%This file is large and may take several minutes to load. We recommend not
%clearing the variable, and only loading it as many times as necessary.

%% Example: removing strides with outliers from kinematic data, and taking the mean

load('gaitModel.mat')

% Gaitcyle and Continuous store the InclineExperiment data in a MATLAB
% structure array, or 'struct'

% The fields of a struct can be iterated over using a cell array

% A cell array can be defined manually with this notation: 
sub={'AB01','AB02','AB03','AB04','AB05','AB06','AB07','AB08','AB09','AB10'};

% sub={'AB05'};

% The command 'fieldnames' also returns the name of all fields within a
% struct as a cell array

% trials = {'s1i10','s1i7x5','s1i5','s1i2x5','s1i0','s1d2x5','s1d5','s1d7x5','s1d10'}
trials = fieldnames(Gaitcycle.AB01);
% The command 'setdiff' is useful for removing cells of a given name
trials = setdiff(trials,'subjectdetails');


leg={'left'};
joint={'foot'};
percent_gait = linspace(0,1,150);

c = {'r','g','b'};
% c1 = {'b','b','b','b','b','r','r','r','r'}
% 
% c2 = [0.5,0.4,0.3,0.2,0.1,0.2,0.3,0.4,0.5]

% Initializing a matrix that will contain the mean joint data of all subjects,
%all tasks
% concatenated_data=NaN(150,numel(sub)*numel(trials));




phaseDelins = [0.1,0.5,0.65,1];

numInclineFuncs = 3;
numStepLengthFuncs = 3;
numPhaseFuncs = 4;
numFuncs = numInclineFuncs*numStepLengthFuncs*numPhaseFuncs;

phase = linspace(0,1,150);

A_covar_mat_master = zeros(150,1);
b_covar_11_master = zeros(150,1);
b_covar_12_master = zeros(150,1);
b_covar_13_master = zeros(150,1);
b_covar_14_master = zeros(150,1);
b_covar_15_master = zeros(150,1);
b_covar_16_master = zeros(150,1);
b_covar_17_master = zeros(150,1);
b_covar_18_master = zeros(150,1);

b_covar_22_master = zeros(150,1);
b_covar_23_master = zeros(150,1);
b_covar_24_master = zeros(150,1);
b_covar_25_master = zeros(150,1);
b_covar_26_master = zeros(150,1);
b_covar_27_master = zeros(150,1);
b_covar_28_master = zeros(150,1);

b_covar_33_master = zeros(150,1);
b_covar_34_master = zeros(150,1);
b_covar_35_master = zeros(150,1);
b_covar_36_master = zeros(150,1);
b_covar_37_master = zeros(150,1);
b_covar_38_master = zeros(150,1);

b_covar_44_master = zeros(150,1);
b_covar_45_master = zeros(150,1);
b_covar_46_master = zeros(150,1);
b_covar_47_master = zeros(150,1);
b_covar_48_master = zeros(150,1);

b_covar_55_master = zeros(150,1);
b_covar_56_master = zeros(150,1);
b_covar_57_master = zeros(150,1);
b_covar_58_master = zeros(150,1);

b_covar_66_master = zeros(150,1);
b_covar_67_master = zeros(150,1);
b_covar_68_master = zeros(150,1);

b_covar_77_master = zeros(150,1);
b_covar_78_master = zeros(150,1);

b_covar_88_master = zeros(150,1);


isPosDef = zeros(150,1);
DO_PELVIS_ZERO = true

foot_angle_means = zeros(1,length(sub));
shank_angle_means = zeros(1,length(sub));
thigh_angle_means = zeros(1,length(sub));
pelvis_angle_means = zeros(1,length(sub));
if DO_PELVIS_ZERO
    for i = 1:length(sub) %loop through all subjects in 'sub'

        switch sub{i}

            case 'AB01'
                footAngleZero = 85.46; %subject 1

            case 'AB02'
                footAngleZero = 85.46 + 4.506 + 1.62; %subject 2

            case 'AB03'
                footAngleZero = 85.46 + 5.564; %subject 3
            case 'AB04'
                footAngleZero = 85.46 + 7.681; %subject 4
            case 'AB05'
                footAngleZero = 85.46 + 5.405; %subject 5
            case 'AB06'
                footAngleZero = 85.46 + 4.089; %subject 6
            case 'AB07'
                footAngleZero = 85.46 + 1.523; %subject 7
            case 'AB08'
                footAngleZero = 85.46 + 3.305; %subject 8
            case 'AB09'
                footAngleZero = 85.46 + 4.396; %subject 9
            case 'AB10'
                footAngleZero = 85.46 + 6.555; %subject 10

        end

            foot_angle_mean = 0;
            shank_angle_mean = 0;
            thigh_angle_mean = 0;
            pelvis_angle_mean = 0;
            N_sum = 0;
            for j = 1:length(trials)

                foot_angle_data = Continuous.(sub{i}).(trials{j}).kinematics.jointangles.(leg{1}).('foot')(:,1);
                foot_angle_data = -(foot_angle_data + footAngleZero);
                ankle_angle_data = Continuous.(sub{i}).(trials{j}).kinematics.jointangles.(leg{1}).('ankle')(:,1);
                hip_angle_data = Continuous.(sub{i}).(trials{j}).kinematics.jointangles.(leg{1}).('hip')(:,1);
                pelvis_angle_data = Continuous.(sub{i}).(trials{j}).kinematics.jointangles.(leg{1}).('pelvis')(:,1);

                shank_angle_data = foot_angle_data - ankle_angle_data;
                thigh_angle_data = pelvis_angle_data + hip_angle_data;

                trial = trials{j};

                N = length(pelvis_angle_data);
                N_sum = N_sum + N;

                foot_angle_mean = foot_angle_mean*((N_sum - N)/N_sum) + (N/N_sum)*mean(foot_angle_data);
                shank_angle_mean = shank_angle_mean*((N_sum - N)/N_sum) + (N/N_sum)*mean(shank_angle_data);
                thigh_angle_mean = thigh_angle_mean*((N_sum - N)/N_sum) + (N/N_sum)*mean(thigh_angle_data);
                pelvis_angle_mean = pelvis_angle_mean*((N_sum - N)/N_sum) + (N/N_sum)*mean(pelvis_angle_data);

        %         pause



            foot_angle_mean;
            shank_angle_mean;
            thigh_angle_mean;
            pelvis_angle_mean;
            end
            
            foot_angle_means(i) = foot_angle_mean;
            shank_angle_means(i) = shank_angle_mean;
            thigh_angle_means(i) = thigh_angle_mean;
            pelvis_angle_means(i) = pelvis_angle_mean;
            
    end
end

shank_angle_means

for phase_idx = 1:150
    
    N = 0;
    X_meas = [];

    
    for i = 1:length(sub) %loop through all subjects in 'sub'
        switch sub{i}

            case 'AB01'
                footAngleZero = 85.46; %subject 1

            case 'AB02'
                footAngleZero = 85.46 + 4.506 + 1.62; %subject 2

            case 'AB03'
                footAngleZero = 85.46 + 5.564; %subject 3
            case 'AB04'
                footAngleZero = 85.46 + 7.681; %subject 4
            case 'AB05'
                footAngleZero = 85.46 + 5.405; %subject 5
            case 'AB06'
                footAngleZero = 85.46 + 4.089; %subject 6
            case 'AB07'
                footAngleZero = 85.46 + 1.523; %subject 7
            case 'AB08'
                footAngleZero = 85.46 + 3.305; %subject 8
            case 'AB09'
                footAngleZero = 85.46 + 4.396; %subject 9
            case 'AB10'
                footAngleZero = 85.46 + 6.555; %subject 10

        end
        
        
        for j = 1:length(trials) %loop through all trials in 'trial'
            % store the kinematic data in a temporary variable
    %         sub{i}
            trial = trials{j};
            foot_angle_data = Gaitcycle.(sub{i}).(trials{j}).kinematics.jointangles.(leg{1}).('foot').x;
            % adjust
            foot_angle_data = -(foot_angle_data + footAngleZero);

            ankle_angle_data = Gaitcycle.(sub{i}).(trials{j}).kinematics.jointangles.(leg{1}).('ankle').x;
            knee_angle_data = Gaitcycle.(sub{i}).(trials{j}).kinematics.jointangles.(leg{1}).('knee').x;
            hip_angle_data = Gaitcycle.(sub{i}).(trials{j}).kinematics.jointangles.(leg{1}).('hip').x;
            pelvis_angle_data = Gaitcycle.(sub{i}).(trials{j}).kinematics.jointangles.(leg{1}).('pelvis').x;

            time_data = Gaitcycle.(sub{i}).(trials{j}).cycles.(leg{1}).time;

            % delete the strides identified to contain outliers
            foot_angle_data(:,Gaitcycle.(sub{i}).(trials{j}).stepsout.(leg{1})) = [];
            ankle_angle_data(:,Gaitcycle.(sub{i}).(trials{j}).stepsout.(leg{1})) = [];
            knee_angle_data(:,Gaitcycle.(sub{i}).(trials{j}).stepsout.(leg{1})) = [];
            hip_angle_data(:,Gaitcycle.(sub{i}).(trials{j}).stepsout.(leg{1})) = [];
            pelvis_angle_data(:,Gaitcycle.(sub{i}).(trials{j}).stepsout.(leg{1})) = [];
            time_data(:,Gaitcycle.(sub{i}).(trials{j}).stepsout.(leg{1})) = [];

            shank_angle_data = foot_angle_data - ankle_angle_data;
            thigh_angle_data = pelvis_angle_data + hip_angle_data;
            
            if DO_PELVIS_ZERO
                foot_angle_data = foot_angle_data - foot_angle_means(i);
                shank_angle_data = shank_angle_data - shank_angle_means(i);
                thigh_angle_data = thigh_angle_data - thigh_angle_means(i);
                pelvis_angle_data = pelvis_angle_data - pelvis_angle_means(i);

            end
        
            
            [treadmillSpeed, treadmillIncline] = returnSpeedIncline(trial);
    %         incline_scaled = (treadmillIncline - inclineMin)/(inclineMax - inclineMin);

            incline_scaled = treadmillIncline;


            

            [rows,cols] = size(foot_angle_data);
            
            for k = 1:cols
                N = N + 1;

                foot_angle_data_col = foot_angle_data(:,k);
                shank_angle_data_col = shank_angle_data(:,k);
                thigh_angle_data_col = thigh_angle_data(:,k);
                pelvis_angle_data_col = pelvis_angle_data(:,k);

                time_data_col = time_data(:,k);
                
                
                % velocities
                foot_vel_data_col = diff(foot_angle_data_col)./diff(time_data_col);
                foot_vel_data_col = [foot_vel_data_col(1);foot_vel_data_col];
                
                shank_vel_data_col = diff(shank_angle_data_col)./diff(time_data_col);
                shank_vel_data_col = [shank_vel_data_col(1);shank_vel_data_col];
                
                thigh_vel_data_col = diff(thigh_angle_data_col)./diff(time_data_col);
                thigh_vel_data_col = [thigh_vel_data_col(1);thigh_vel_data_col];
                
                pelvis_vel_data_col = diff(pelvis_angle_data_col)./diff(time_data_col);
                pelvis_vel_data_col = [pelvis_vel_data_col(1);pelvis_vel_data_col];

            
            

                stepDuration = time_data_col(end) - time_data_col(1);
                phaseDot = 1/stepDuration;
                stepLength = treadmillSpeed*stepDuration;

                stepLength_scaled = stepLength;


                phase_col = time_data_col/stepDuration;
                phase_j = phase_col(phase_idx);
                
                footAngle_measured = foot_angle_data_col(phase_idx);
                shankAngle_measured = shank_angle_data_col(phase_idx);
                thighAngle_measured = thigh_angle_data_col(phase_idx);
                pelvisAngle_measured = pelvis_angle_data_col(phase_idx);
                
                footAngleVel_measured = foot_vel_data_col(phase_idx);
                shankAngleVel_measured = shank_vel_data_col(phase_idx);
                thighAngleVel_measured = thigh_vel_data_col(phase_idx);
                pelvisAngleVel_measured = pelvis_vel_data_col(phase_idx);

                footAngle_model = returnPiecewiseBezier3D(phase_j,stepLength_scaled,incline_scaled,best_fit_params_footAngle, phaseDelins,numFuncs);
                shankAngle_model = returnPiecewiseBezier3D(phase_j,stepLength_scaled,incline_scaled,best_fit_params_shankAngle, phaseDelins,numFuncs);
                thighAngle_model = returnPiecewiseBezier3D(phase_j,stepLength_scaled,incline_scaled,best_fit_params_thighAngle, phaseDelins,numFuncs);
                pelvisAngle_model = returnPiecewiseBezier3D(phase_j,stepLength_scaled,incline_scaled,best_fit_params_pelvisAngle, phaseDelins,numFuncs);

                footAngleVel_model = phaseDot * returnPiecewiseBezier3DDeriv_dphase(phase_j,stepLength_scaled,incline_scaled,best_fit_params_footAngle, phaseDelins,numFuncs);
                shankAngleVel_model = phaseDot * returnPiecewiseBezier3DDeriv_dphase(phase_j,stepLength_scaled,incline_scaled,best_fit_params_shankAngle, phaseDelins,numFuncs);
                thighAngleVel_model = phaseDot * returnPiecewiseBezier3DDeriv_dphase(phase_j,stepLength_scaled,incline_scaled,best_fit_params_thighAngle, phaseDelins,numFuncs);
                pelvisAngleVel_model = phaseDot * returnPiecewiseBezier3DDeriv_dphase(phase_j,stepLength_scaled,incline_scaled,best_fit_params_pelvisAngle, phaseDelins,numFuncs);


                footAngle_residual = footAngle_measured - footAngle_model;
                shankAngle_residual = shankAngle_measured - shankAngle_model;
                thighAngle_residual = thighAngle_measured - thighAngle_model;
                pelvisAngle_residual = pelvisAngle_measured - pelvisAngle_model;
                
                footAngleVel_residual = footAngleVel_measured - footAngleVel_model;
                shankAngleVel_residual = shankAngleVel_measured - shankAngleVel_model;
                thighAngleVel_residual = thighAngleVel_measured - thighAngleVel_model;
                pelvisAngleVel_residual = pelvisAngleVel_measured - pelvisAngleVel_model;
                
                X_meas = [X_meas;...
                    [footAngle_residual,footAngleVel_residual,...
                    shankAngle_residual,shankAngleVel_residual,...
                    thighAngle_residual,thighAngleVel_residual,...
                    pelvisAngle_residual,pelvisAngleVel_residual];];



            end

        end
    end
    X_meas;
    N;
    C = ( 1./(N - 1)) * (X_meas' * X_meas);
%     pause
    
    C = C + 1e-6*eye(8);
    
    
%     pause
    
    
    b_covar_11_master(phase_idx) = C(1,1);
    b_covar_12_master(phase_idx) = C(1,2);
    b_covar_13_master(phase_idx) = C(1,3);
    b_covar_14_master(phase_idx) = C(1,4);
    b_covar_15_master(phase_idx) = C(1,5);
    b_covar_16_master(phase_idx) = C(1,6);
    b_covar_17_master(phase_idx) = C(1,7);
    b_covar_18_master(phase_idx) = C(1,8);
    
    b_covar_22_master(phase_idx) = C(2,2);
    b_covar_23_master(phase_idx) = C(2,3);
    b_covar_24_master(phase_idx) = C(2,4);
    b_covar_25_master(phase_idx) = C(2,5);
    b_covar_26_master(phase_idx) = C(2,6);
    b_covar_27_master(phase_idx) = C(2,7);
    b_covar_28_master(phase_idx) = C(2,8);
    
    b_covar_33_master(phase_idx) = C(3,3);
    b_covar_34_master(phase_idx) = C(3,4);
    b_covar_35_master(phase_idx) = C(3,5);
    b_covar_36_master(phase_idx) = C(3,6);
    b_covar_37_master(phase_idx) = C(3,7);
    b_covar_38_master(phase_idx) = C(3,8);
    
    b_covar_44_master(phase_idx) = C(4,4);
    b_covar_45_master(phase_idx) = C(4,5);
    b_covar_46_master(phase_idx) = C(4,6);
    b_covar_47_master(phase_idx) = C(4,7);
    b_covar_48_master(phase_idx) = C(4,8);
    
    b_covar_55_master(phase_idx) = C(5,5);
    b_covar_56_master(phase_idx) = C(5,6);
    b_covar_57_master(phase_idx) = C(5,7);
    b_covar_58_master(phase_idx) = C(5,8);
    
    b_covar_66_master(phase_idx) = C(6,6);
    b_covar_67_master(phase_idx) = C(6,7);
    b_covar_68_master(phase_idx) = C(6,8);
    
    b_covar_77_master(phase_idx) = C(7,7);
    b_covar_78_master(phase_idx) = C(7,8);
    
    b_covar_88_master(phase_idx) = C(8,8);
    
    [~,p] = chol(C);
    
    isPosDef(phase_idx) = p == 0;

end


%%

% b_covar_11_master = lowpass(b_covar_11_master,10,120);
% b_covar_12_master = lowpass(b_covar_12_master,10,120);
% b_covar_13_master = lowpass(b_covar_13_master,10,120);
% b_covar_14_master = lowpass(b_covar_14_master,10,120);
% b_covar_15_master = lowpass(b_covar_15_master,10,120);
% b_covar_16_master = lowpass(b_covar_16_master,10,120);
% 
% b_covar_22_master = lowpass(b_covar_22_master,10,120);
% b_covar_23_master = lowpass(b_covar_23_master,10,120);
% b_covar_24_master = lowpass(b_covar_24_master,10,120);
% b_covar_25_master = lowpass(b_covar_25_master,10,120);
% b_covar_26_master = lowpass(b_covar_26_master,10,120);
% 
% b_covar_33_master = lowpass(b_covar_33_master,10,120);
% b_covar_34_master = lowpass(b_covar_34_master,10,120);
% b_covar_35_master = lowpass(b_covar_35_master,10,120);
% b_covar_36_master = lowpass(b_covar_36_master,10,120);
% 
% b_covar_44_master = lowpass(b_covar_44_master,10,120);
% b_covar_45_master = lowpass(b_covar_45_master,10,120);
% b_covar_46_master = lowpass(b_covar_46_master,10,120);
% 
% b_covar_55_master = lowpass(b_covar_55_master,10,120);
% b_covar_56_master = lowpass(b_covar_56_master,10,120);
% 
% b_covar_66_master = lowpass(b_covar_66_master,10,120);



figure(1)
subplot(8,1,1)
semilogy(phase, b_covar_11_master,'LineWidth',2)
hold on
legend('11')


subplot(8,1,2)
semilogy(phase, b_covar_22_master,'LineWidth',2)
hold on
legend('22')

subplot(8,1,3)
semilogy(phase, b_covar_33_master,'LineWidth',2)
hold on
legend('33')


subplot(8,1,4)
semilogy(phase, b_covar_44_master,'LineWidth',2)
hold on
legend('44')

subplot(8,1,5)
semilogy(phase, b_covar_55_master,'LineWidth',2)
hold on
legend('55')

subplot(8,1,6)
semilogy(phase, b_covar_66_master,'LineWidth',2)
hold on
legend('66')

subplot(8,1,7)
semilogy(phase, b_covar_77_master,'LineWidth',2)
hold on
legend('77')

subplot(8,1,8)
semilogy(phase, b_covar_78_master,'LineWidth',2)
hold on
legend('88')

figure(2)

subplot(8,1,1)
plot(phase, b_covar_11_master,'LineWidth',2)
hold on
legend('11')


subplot(8,1,2)
plot(phase, b_covar_12_master,'LineWidth',2)
hold on
legend('12')

subplot(8,1,3)
plot(phase, b_covar_13_master,'LineWidth',2)
hold on
legend('13')


subplot(8,1,4)
plot(phase, b_covar_14_master,'LineWidth',2)
hold on
legend('14')

subplot(8,1,5)
plot(phase, b_covar_15_master,'LineWidth',2)
hold on
legend('15')

subplot(8,1,6)
plot(phase, b_covar_16_master,'LineWidth',2)
hold on
legend('16')

subplot(8,1,7)
plot(phase, b_covar_17_master,'LineWidth',2)
hold on
legend('17')

subplot(8,1,8)
plot(phase, b_covar_18_master,'LineWidth',2)
hold on
legend('18')


figure(3)

subplot(5,1,1)
plot(phase, b_covar_22_master,'LineWidth',2)
hold on
legend('22')


subplot(5,1,2)
plot(phase, b_covar_23_master,'LineWidth',2)
hold on
legend('23')

subplot(5,1,3)
plot(phase, b_covar_24_master,'LineWidth',2)
hold on
legend('24')

subplot(5,1,4)
plot(phase, b_covar_25_master,'LineWidth',2)
hold on
legend('25')

subplot(5,1,5)
plot(phase, b_covar_26_master,'LineWidth',2)
hold on
legend('26')



figure(4)

subplot(4,1,1)
plot(phase, b_covar_33_master,'LineWidth',2)
hold on
legend('33')


subplot(4,1,2)
plot(phase, b_covar_34_master,'LineWidth',2)
hold on
legend('34')

subplot(4,1,3)
plot(phase, b_covar_35_master,'LineWidth',2)
hold on
legend('35')

subplot(4,1,4)
plot(phase, b_covar_36_master,'LineWidth',2)
hold on
legend('36')


figure(5)

subplot(3,1,1)
plot(phase, b_covar_44_master,'LineWidth',2)
hold on
legend('44')


subplot(3,1,2)
plot(phase, b_covar_45_master,'LineWidth',2)
hold on
legend('45')

subplot(3,1,3)
plot(phase, b_covar_46_master,'LineWidth',2)
hold on
legend('46')


figure(6)

subplot(2,1,1)
plot(phase, b_covar_55_master,'LineWidth',2)
hold on
legend('55')


subplot(2,1,2)
plot(phase, b_covar_56_master,'LineWidth',2)
hold on
legend('56')





figure(200)
plot(phase, isPosDef,'LineWidth',2)
legend('PD test')




%%
M = [b_covar_11_master';b_covar_12_master';b_covar_13_master';b_covar_14_master';b_covar_15_master';b_covar_16_master';b_covar_17_master';b_covar_18_master';...
    b_covar_22_master';b_covar_23_master';b_covar_24_master';b_covar_25_master';b_covar_26_master';b_covar_27_master';b_covar_28_master';...
    b_covar_33_master';b_covar_34_master';b_covar_35_master';b_covar_36_master';b_covar_37_master';b_covar_38_master';...
    b_covar_44_master';b_covar_45_master';b_covar_46_master';b_covar_47_master';b_covar_48_master';...
    b_covar_55_master';b_covar_56_master';b_covar_57_master';b_covar_58_master';...
    b_covar_66_master';b_covar_67_master';b_covar_68_master';...
    b_covar_77_master';b_covar_78_master';...
    b_covar_88_master';...
    ];
writematrix(M,'gaitModel_covars.csv')

%% Helper function
function [speed, incline] = returnSpeedIncline(trialString)



d_idx = strfind(trialString,'d');
i_idx = strfind(trialString,'i');

isIncline = ~isempty(i_idx) & isempty(d_idx);
isDecline = isempty(i_idx) & ~isempty(d_idx);

mid_idx = [i_idx,d_idx];

speedString = trialString(2:mid_idx-1);

inclineString = trialString(mid_idx+1:end);



switch speedString
            
    case '0x8'
        speed = 0.8;

    case '1'
        speed = 1;

    case '1x2'
        speed = 1.2;

end


switch inclineString
            
    case '10'
        incline = 10;

    case '7x5'
        incline = 7.5;

    case '5'
        incline = 5;
        
    case '2x5'
        incline = 2.5;
        
    case '0'
        incline = 0;
        

end
if isDecline
    incline = incline * -1;
end



end
