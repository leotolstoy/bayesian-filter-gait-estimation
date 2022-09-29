close all;clc;clearvars -except Gaitcycle Continuous %removes all variables except for gaitcyle and continuous from the matlab workspace. Useful because loading these variables can take several minutes.
%load the InclineExperiment data from the folder location you specify
% load('Z:\your_file_location_here\InclineExperiment.mat') 
%This file is large and may take several minutes to load. We recommend not
%clearing the variable, and only loading it as many times as necessary.

%% Example: removing strides with outliers from kinematic data, and taking the mean

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

c = {'r','g','b'}
% c1 = {'b','b','b','b','b','r','r','r','r'}
% 
% c2 = [0.5,0.4,0.3,0.2,0.1,0.2,0.3,0.4,0.5]

% Initializing a matrix that will contain the mean joint data of all subjects,
%all tasks
% concatenated_data=NaN(150,numel(sub)*numel(trials));

% figure(1)
% subplot(1,3,1)
% hold on
% subplot(1,3,2)
% hold on
% subplot(1,3,3)
% hold on

figure(2)
subplot(1,3,1)
hold on
subplot(1,3,2)
hold on
subplot(1,3,3)
hold on





% figure(3)
% subplot(1,3,1)
% hold on
% subplot(1,3,2)
% hold on
% subplot(1,3,3)
% hold on


figure(4)
subplot(1,3,1)
hold on
subplot(1,3,2)
hold on
subplot(1,3,3)
hold on


figure(39)
subplot(1,3,1)
hold on
subplot(1,3,2)
hold on
subplot(1,3,3)
hold on

figure(40)
subplot(1,3,1)
hold on
subplot(1,3,2)
hold on
subplot(1,3,3)
hold on


figure(7)
hold on

figure(8)
hold on


phaseDelins = [0.1,0.5,0.65,1]

numInclineFuncs = 3;
numStepLengthFuncs = 3;
numPhaseFuncs = 4;
numFuncs = numInclineFuncs*numStepLengthFuncs*numPhaseFuncs;

A_dataport_mat_master = [];
b_dataport_footAngle_master = [];
b_dataport_shankAngle_master = [];
b_dataport_thighAngle_master = [];
b_dataport_pelvisAngle_master = [];

DOEACHSTEPPLOT = false
DOAVGSTEPPLOT = false
DO_PELVIS_ZERO = true
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
    
    if DO_PELVIS_ZERO
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

        end


        shank_angle_mean
        thigh_angle_mean
        pelvis_angle_mean
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
            foot_angle_data = foot_angle_data - foot_angle_mean;
            shank_angle_data = shank_angle_data - shank_angle_mean;
            thigh_angle_data = thigh_angle_data - thigh_angle_mean;
            pelvis_angle_data = pelvis_angle_data - pelvis_angle_mean;
            
        end
        [treadmillSpeed, treadmillIncline] = returnSpeedIncline(trial);
        
        incline_scaled = treadmillIncline;

        
        
        [rows,cols] = size(foot_angle_data);
        
        A_dataport_mat = zeros(cols * 150,4*numFuncs);
        b_dataport_footAngle_mat = zeros(cols * 150,1);
        b_dataport_shankAngle_mat = zeros(cols * 150,1);
        b_dataport_thighAngle_mat = zeros(cols * 150,1);
        b_dataport_pelvisAngle_mat = zeros(cols * 150,1);
        
%         stepLengths = zeros(cols,1);
        
        stepDurations = time_data(end,:) - time_data(1,:);
        stepLengths = treadmillSpeed.*stepDurations;
        
        for k = 1:cols
            
            foot_angle_data_col = foot_angle_data(:,k);
            shank_angle_data_col = shank_angle_data(:,k);
            thigh_angle_data_col = thigh_angle_data(:,k);
            pelvis_angle_data_col = pelvis_angle_data(:,k);
            
            
            time_data_col = time_data(:,k);
            
            stepDuration = time_data_col(end) - time_data_col(1);
            phaseDot = 1/stepDuration;
            stepLength = treadmillSpeed*stepDuration;
                        
            stepLength_scaled = stepLength;
            
            stepLengths(k) = stepLength_scaled;

            
            phase_col = time_data_col/stepDuration;
            
            if (strcmp(sub{i}, 'AB01') || length(sub) == 1) && DOEACHSTEPPLOT
                
                if treadmillIncline == 10
                    subplotidx = 1;
                elseif treadmillIncline == 0
                    subplotidx = 2;
                elseif treadmillIncline == -10
                    subplotidx = 3;
                end
         
                % plot close enough stride lengths against the ramp surface
                
                if abs(stepLength_scaled - 1) <= 0.1 && (treadmillIncline == 10 || treadmillIncline == 0 || treadmillIncline == -10)
                    figure(7)
                    plot3(phase_col,treadmillIncline*ones(size(phase_col)), thigh_angle_data_col,'LineWidth',2)
                    figure(8)
                    plot3(phase_col,treadmillIncline*ones(size(phase_col)), pelvis_angle_data_col,'LineWidth',2)
                end
                
            end
            
            
            
            for phase_idx = 1:length(phase_col)
                mat_idx = phase_idx + (k - 1)*150;
                phase_j = phase_col(phase_idx);

                bezier_coeffs = returnBezierEval3P_sL_incline(phase_j,stepLength_scaled,incline_scaled);

                if phase_j <= phaseDelins(1)
                    bezier_row = [bezier_coeffs, zeros(1,3*numFuncs)];

                elseif phase_j <= phaseDelins(2)
                    bezier_row = [zeros(1,1*numFuncs),bezier_coeffs, zeros(1,2*numFuncs)];
                elseif phase_j <= phaseDelins(3)
                    bezier_row = [zeros(1,2*numFuncs),bezier_coeffs, zeros(1,1*numFuncs)];
                elseif phase_j <= phaseDelins(4)
                    bezier_row = [zeros(1,3*numFuncs),bezier_coeffs];

                end

                A_dataport_mat(mat_idx,:) = bezier_row;
                b_dataport_footAngle_mat(mat_idx) = foot_angle_data_col(phase_idx);
                b_dataport_shankAngle_mat(mat_idx) = shank_angle_data_col(phase_idx);
                b_dataport_thighAngle_mat(mat_idx) = thigh_angle_data_col(phase_idx);
                b_dataport_pelvisAngle_mat(mat_idx) = pelvis_angle_data_col(phase_idx);
                
        
        
            end
            
    
            
            
        end
        A_dataport_mat_master = [A_dataport_mat_master;A_dataport_mat];
        b_dataport_footAngle_master = [b_dataport_footAngle_master;b_dataport_footAngle_mat];
        b_dataport_shankAngle_master = [b_dataport_shankAngle_master;b_dataport_shankAngle_mat];
        b_dataport_thighAngle_master = [b_dataport_thighAngle_master;b_dataport_thighAngle_mat];
        b_dataport_pelvisAngle_master = [b_dataport_pelvisAngle_master;b_dataport_pelvisAngle_mat];
        
        avgStepDuration = mean(time_data(end,:) - time_data(1,:) );
        avgFootAngle = mean(foot_angle_data,2);
        avgShankAngle = mean(shank_angle_data,2);
        avgThighAngle = mean(thigh_angle_data,2);
        avgPelvisAngle = mean(pelvis_angle_data,2);
        avgPhase = percent_gait';
        
        avgStepLength = mean(stepLengths);
        
        
        if (strcmp(sub{i}, 'AB01') || length(sub) == 1 || true) && DOAVGSTEPPLOT

            
            if treadmillIncline == 10
                subplotidx = 1;
            elseif treadmillIncline == 0
                subplotidx = 2;
            elseif treadmillIncline == -10
                subplotidx = 3;
            end
            
            if treadmillSpeed == 1.2
                cl = 'r';
            elseif treadmillSpeed == 1
                cl = 'b';
            elseif treadmillSpeed == 0.8
                cl = 'g';
            end
            
            
            if treadmillIncline == 10 || treadmillIncline == 0 || treadmillIncline == -10
                figure(2)
                subplot(1,3,subplotidx)
                plot_j = plot3(avgPhase,avgStepLength*ones(size(avgPhase)), avgFootAngle, cl,'LineWidth',2);
                plot_j.Color(4) = 1;
                if DOEACHSTEPPLOT
                    plot_j = plot3(repmat(phase_col,1,cols),repmat(stepLengths,150,1), foot_angle_data,'LineWidth',2);
                end
                figure(4)
                subplot(1,3,subplotidx)
                plot_j = plot3(avgPhase,avgStepLength*ones(size(avgPhase)), avgShankAngle, cl,'LineWidth',2);
                plot_j.Color(4) = 1;
                if DOEACHSTEPPLOT
                    plot_j = plot3(repmat(phase_col,1,cols),repmat(stepLengths,150,1), shank_angle_data,'LineWidth',2);
                end
                figure(39)
                subplot(1,3,subplotidx)
                plot_j = plot3(avgPhase,avgStepLength*ones(size(avgPhase)), avgThighAngle, cl,'LineWidth',2);
                plot_j.Color(4) = 1;
                if DOEACHSTEPPLOT
                    plot_j = plot3(repmat(phase_col,1,cols),repmat(stepLengths,150,1), thigh_angle_data,'LineWidth',2);
                end
                figure(40)
                subplot(1,3,subplotidx)
                plot_j = plot3(avgPhase,avgStepLength*ones(size(avgPhase)), avgPelvisAngle, cl,'LineWidth',2);
                plot_j.Color(4) = 1;
                if DOEACHSTEPPLOT
                    plot_j = plot3(repmat(phase_col,1,cols),repmat(stepLengths,150,1), pelvis_angle_data,'LineWidth',2);
                end
            end


            
            
        end
%         pause

        
        
        
%         pause
        
        % take the mean of the remaining strides and concatenate them
        
        
%         nanmean(temp_data,2)
%         concatenated_data(:, numel(trials)*(s-1)+t ) = nanmean(temp_data,2);
    end
end

% The data in Gaitcyle always has 150 points per stride, and they are
% evenly spaced with respect to percent gait.

% figure(1)
% title([leg{1}, ' ', joint{1}, ' joint angles'])
% subplot(1,3,1)
% 
% xlabel('percent gait')
% % ylabel('Step Length (m)')
% ylabel('degrees'); 
% 
% subplot(1,3,2)
% 
% % ylabel('Step Length (m)')
% xlabel('step Length'); 
% ylabel('degrees'); 
% 
% subplot(1,3,3)
% 
% % ylabel('Step Length (m)')
% xlabel('phase_dot'); 
% ylabel('degrees'); 


figure(2)
title([leg{1}, ' ', joint{1}, ' joint angles'])
xlabel('percent gait')
ylabel('Step Length (1/s)')
zlabel('degrees'); 
view(60,60)

subplot(1,3,1)
title('10 deg incline')
view(60,60)
subplot(1,3,2)
title('0 deg incline')
view(60,60)
subplot(1,3,3)
title('-10 deg incline')
view(60,60)
% figure(3)
% title([leg{1}, ' ', 'shank', ' joint angles'])
% subplot(1,3,1)
% 
% xlabel('percent gait')
% % ylabel('Step Length (m)')
% ylabel('degrees'); 
% 
% subplot(1,3,2)
% 
% % ylabel('Step Length (m)')
% xlabel('step Length'); 
% ylabel('degrees'); 
% 
% subplot(1,3,3)
% 
% % ylabel('Step Length (m)')
% xlabel('phase_dot'); 
% ylabel('degrees'); 


figure(4)
title([leg{1}, ' ', 'shank', ' joint angles'])
xlabel('percent gait')
ylabel('Step Length (m)')
zlabel('degrees'); 
view(60,60)

subplot(1,3,1)
title('10 deg incline')
view(60,60)
subplot(1,3,2)
title('0 deg incline')
view(60,60)
subplot(1,3,3)
title('-10 deg incline')
view(60,60)


figure(39)
title([leg{1}, ' ', 'thigh', ' joint angles'])
xlabel('percent gait')
ylabel('Step Length (m)')
zlabel('degrees'); 
view(60,60)

subplot(1,3,1)
title('10 deg incline')
view(60,60)
subplot(1,3,2)
title('0 deg incline')
view(60,60)
subplot(1,3,3)
title('-10 deg incline')
view(60,60)


figure(40)
title([leg{1}, ' ', 'pelvis', ' joint angles'])
xlabel('percent gait')
ylabel('Step Length (m)')
zlabel('degrees'); 
view(60,60)

subplot(1,3,1)
title('10 deg incline')
view(60,60)
subplot(1,3,2)
title('0 deg incline')
view(60,60)
subplot(1,3,3)
title('-10 deg incline')
view(60,60)

%% Regress

%C0 and %C1 continuity constraints
stride_lengths = linspace(0,2,3);
% stride_lengths = [0,1];
% ramps = [0,1];
ramps = linspace(-10,10,3);
A_eq_C0_constraint = [];
A_eq_C1_constraint = [];
for ramp = ramps
    
    A_eq_C0_ramp = [];
    A_eq_C1_ramp = [];
    for j= 1:length(stride_lengths)
        stride_length = stride_lengths(j);
        
        A_eq_bezier1 = [[returnBezierEval3P_sL_incline(phaseDelins(1),stride_length,ramp), -returnBezierEval3P_sL_incline(phaseDelins(1),stride_length,ramp), zeros(1,numFuncs), zeros(1,numFuncs)];...
        [returnBezier3P_sL_incline_DerivEval_dphase(phaseDelins(1),stride_length,ramp), -returnBezier3P_sL_incline_DerivEval_dphase(phaseDelins(1),stride_length,ramp), zeros(1,numFuncs), zeros(1,numFuncs)];...
        [zeros(1,numFuncs), returnBezierEval3P_sL_incline(phaseDelins(2),stride_length,ramp), -returnBezierEval3P_sL_incline(phaseDelins(2),stride_length,ramp), zeros(1,numFuncs)];...
        [zeros(1,numFuncs), returnBezier3P_sL_incline_DerivEval_dphase(phaseDelins(2),stride_length,ramp), -returnBezier3P_sL_incline_DerivEval_dphase(phaseDelins(2),stride_length,ramp), zeros(1,numFuncs)];...
        [zeros(1,numFuncs), zeros(1,numFuncs), returnBezierEval3P_sL_incline(phaseDelins(3),stride_length,ramp), -returnBezierEval3P_sL_incline(phaseDelins(3),stride_length,ramp)];...
        [zeros(1,numFuncs), zeros(1,numFuncs), returnBezier3P_sL_incline_DerivEval_dphase(phaseDelins(3),stride_length,ramp), -returnBezier3P_sL_incline_DerivEval_dphase(phaseDelins(3),stride_length,ramp)];...
        [-returnBezierEval3P_sL_incline(0,stride_length,ramp), zeros(1,numFuncs), zeros(1,numFuncs), returnBezierEval3P_sL_incline(phaseDelins(4),stride_length,ramp)];...
        [-returnBezier3P_sL_incline_DerivEval_dphase(0,stride_length,ramp), zeros(1,numFuncs), zeros(1,numFuncs), returnBezier3P_sL_incline_DerivEval_dphase(phaseDelins(4),stride_length,ramp)]];


        A_eq_bezier2 = [[returnBezierEval3P_sL_incline(phaseDelins(1),stride_length,ramp), -returnBezierEval3P_sL_incline(phaseDelins(1),stride_length,ramp), zeros(1,numFuncs), zeros(1,numFuncs)];...
        [returnBezier3P_sL_incline_DerivEval_dphase(phaseDelins(1),stride_length,ramp), -returnBezier3P_sL_incline_DerivEval_dphase(phaseDelins(1),stride_length,ramp), zeros(1,numFuncs), zeros(1,numFuncs)];...
        [zeros(1,numFuncs), returnBezierEval3P_sL_incline(phaseDelins(2),stride_length,ramp), -returnBezierEval3P_sL_incline(phaseDelins(2),stride_length,ramp), zeros(1,numFuncs)];...
        [zeros(1,numFuncs), returnBezier3P_sL_incline_DerivEval_dphase(phaseDelins(2),stride_length,ramp), -returnBezier3P_sL_incline_DerivEval_dphase(phaseDelins(2),stride_length,ramp), zeros(1,numFuncs)];...
        [zeros(1,numFuncs), zeros(1,numFuncs), returnBezierEval3P_sL_incline(phaseDelins(3),stride_length,ramp), -returnBezierEval3P_sL_incline(phaseDelins(3),stride_length,ramp)];...
        [zeros(1,numFuncs), zeros(1,numFuncs), returnBezier3P_sL_incline_DerivEval_dphase(phaseDelins(3),stride_length,ramp), -returnBezier3P_sL_incline_DerivEval_dphase(phaseDelins(3),stride_length,ramp)];...
        [-returnBezierEval3P_sL_incline(0,stride_length,ramp), zeros(1,numFuncs), zeros(1,numFuncs), returnBezierEval3P_sL_incline(phaseDelins(4),stride_length,ramp)];...
        [-returnBezier3P_sL_incline_DerivEval_dphase(0,stride_length,ramp), zeros(1,numFuncs), zeros(1,numFuncs), returnBezier3P_sL_incline_DerivEval_dphase(phaseDelins(4),stride_length,ramp)]];



        A_eq_C0_ramp = [A_eq_C0_ramp;A_eq_bezier1];
        A_eq_C1_ramp = [A_eq_C1_ramp;A_eq_bezier2];

    end
    
    A_eq_C0_constraint = [A_eq_C0_constraint;A_eq_C0_ramp];
    A_eq_C1_constraint = [A_eq_C1_constraint;A_eq_C1_ramp];

end   

[rows,~] = size(A_eq_C0_constraint);
b_eq_C0_constraint = zeros(rows,1);

[rows,~] = size(A_eq_C1_constraint);
b_eq_C1_constraint = zeros(rows,1);


%% force constant behavior at sL = 0


phase = linspace(0,1,3*4)';



A_eq_bezier_shank_constAtZerosL = [];
ramps = linspace(-10,10,3);
% ramps = [0,10,-10];
for ramp = ramps
    
    A_eq_bezier_shank_ramp = zeros(length(phase), 4*numFuncs);
    b_eq_shank_ramp = zeros(length(phase), 1);
    
    bezier_deriv_coeffs0 = returnBezierEval3P_sL_incline(0,0,ramp);
    
    for j= 1:length(phase)
        
        
        
        phase_j = phase(j);
        
        bezier_deriv_coeffs = returnBezierEval3P_sL_incline(phase_j,0,ramp);

        if phase_j <= phaseDelins(1)
            bezier_row = [bezier_deriv_coeffs - bezier_deriv_coeffs0, zeros(1,3*numFuncs)];
        elseif phase_j <= phaseDelins(2)
            bezier_row = [-bezier_deriv_coeffs0,bezier_deriv_coeffs, zeros(1,2*numFuncs)];
        elseif phase_j <= phaseDelins(3)
            bezier_row = [-bezier_deriv_coeffs0, zeros(1,1*numFuncs),bezier_deriv_coeffs, zeros(1,1*numFuncs)];
        elseif phase_j <= phaseDelins(4)
            bezier_row = [-bezier_deriv_coeffs0, zeros(1,2*numFuncs),bezier_deriv_coeffs];

        end

        A_eq_bezier_shank_ramp(j,:) = bezier_row;

    end
    
    
    A_eq_bezier_shank_constAtZerosL = [A_eq_bezier_shank_constAtZerosL;A_eq_bezier_shank_ramp];

end     


                
A_eq_bezier_shank_zerodsLatzerosL = [];
ramps = linspace(-10,10,3);
% ramps = [0,10,-10];
for ramp = ramps
    
    A_eq_bezier_shank_ramp = zeros(length(phase), 4*numFuncs);
    for j= 1:length(phase)
        phase_j = phase(j);

        bezier_deriv_coeffs = returnBezier3P_sL_incline_DerivEval_dsL(phase_j,0,ramp);

        if phase_j <= phaseDelins(1)
            bezier_row = [bezier_deriv_coeffs, zeros(1,3*numFuncs)];

        elseif phase_j <= phaseDelins(2)
            bezier_row = [zeros(1,1*numFuncs),bezier_deriv_coeffs, zeros(1,2*numFuncs)];
        elseif phase_j <= phaseDelins(3)
            bezier_row = [zeros(1,2*numFuncs),bezier_deriv_coeffs, zeros(1,1*numFuncs)];
        elseif phase_j <= phaseDelins(4)
            bezier_row = [zeros(1,3*numFuncs),bezier_deriv_coeffs];

        end

        A_eq_bezier_shank_ramp(j,:) = bezier_row;

    end
    
    A_eq_bezier_shank_zerodsLatzerosL = [A_eq_bezier_shank_zerodsLatzerosL;A_eq_bezier_shank_ramp];

end   


A_eq_bezier_shank_zerodinclineatzerosL = [];
ramps = linspace(-10,10,3);
% ramps = [0,10,-10];
for ramp = ramps
    
    A_eq_bezier_shank_ramp = zeros(length(phase), 4*numFuncs);
    for j= 1:length(phase)
        phase_j = phase(j);

        bezier_deriv_coeffs = returnBezier3P_sL_incline_DerivEval_dsL(phase_j,0,ramp);

        if phase_j <= phaseDelins(1)
            bezier_row = [bezier_deriv_coeffs, zeros(1,3*numFuncs)];

        elseif phase_j <= phaseDelins(2)
            bezier_row = [zeros(1,1*numFuncs),bezier_deriv_coeffs, zeros(1,2*numFuncs)];
        elseif phase_j <= phaseDelins(3)
            bezier_row = [zeros(1,2*numFuncs),bezier_deriv_coeffs, zeros(1,1*numFuncs)];
        elseif phase_j <= phaseDelins(4)
            bezier_row = [zeros(1,3*numFuncs),bezier_deriv_coeffs];

        end

        A_eq_bezier_shank_ramp(j,:) = bezier_row;

    end
    
    A_eq_bezier_shank_zerodinclineatzerosL = [A_eq_bezier_shank_zerodinclineatzerosL;A_eq_bezier_shank_ramp];

end  


A_eq_bezier_shank_constdsL = [];
ramps = linspace(-10,10,3);
% ramps = [0,10,-10];
for ramp = ramps
    
    bezier_deriv_coeffs0 = returnBezier3P_sL_incline_DerivEval_dsL(0,1,ramp);
    
    A_eq_bezier_shank_ramp = zeros(length(phase), 4*numFuncs);
    for j= 1:length(phase)
        phase_j = phase(j);

        bezier_deriv_coeffs = returnBezier3P_sL_incline_DerivEval_dsL(phase_j,0,ramp);

        if phase_j <= phaseDelins(1)
            bezier_row = [bezier_deriv_coeffs - bezier_deriv_coeffs0, zeros(1,3*numFuncs)];

        elseif phase_j <= phaseDelins(2)
            bezier_row = [-bezier_deriv_coeffs0, bezier_deriv_coeffs, zeros(1,2*numFuncs)];
        elseif phase_j <= phaseDelins(3)
            bezier_row = [-bezier_deriv_coeffs0, zeros(1,1*numFuncs),bezier_deriv_coeffs, zeros(1,1*numFuncs)];
        elseif phase_j <= phaseDelins(4)
            bezier_row = [-bezier_deriv_coeffs0, zeros(1,2*numFuncs),bezier_deriv_coeffs];

        end

        A_eq_bezier_shank_ramp(j,:) = bezier_row;

    end
    
    A_eq_bezier_shank_constdsL = [A_eq_bezier_shank_constdsL;A_eq_bezier_shank_ramp];

end   

% A_eq_foot = [A_eq_C0_constraint;A_eq_C1_constraint;A_eq_bezier_footL1;A_eq_bezier_footL2;A_eq_bezier_footL4];
% A_eq_shank = [A_eq_C0_constraint;A_eq_C1_constraint;A_eq_bezier_shankL1;A_eq_bezier_shankL2];
% 
% b_eq_foot1 = zeros(8,1)
% b_eq_foot2 = zeros(8,1)
% b_eq_foot2(1:2:8) = 10 
% b_eq_foot3 = [10;0;-10];
% b_eq_foot4 = zeros(1,3);
% 


b_eq_shank1 = zeros(4,1);
b_eq_shank2 = zeros(4,1);

[rows,~] = size(A_eq_bezier_shank_constAtZerosL);
b_eq_shank_constAtZerosL = zeros(rows,1);

[rows,~] = size(A_eq_bezier_shank_zerodsLatzerosL);
b_eq_shank_zerodsLatzerosL = zeros(rows,1);

[rows,~] = size(A_eq_bezier_shank_zerodinclineatzerosL);
b_eq_shank_zerodinclineatzerosL = zeros(rows,1);

[rows,~] = size(A_eq_bezier_shank_constdsL);
b_eq_shank_constdsL = zeros(rows,1);

A_eq_foot = [A_eq_C0_constraint;A_eq_C1_constraint;A_eq_bezier_shank_constAtZerosL;A_eq_bezier_shank_zerodsLatzerosL;];
A_eq_shank = [A_eq_C0_constraint;A_eq_C1_constraint;A_eq_bezier_shank_constAtZerosL;A_eq_bezier_shank_zerodsLatzerosL;];
A_eq_thigh = [A_eq_C0_constraint;A_eq_C1_constraint;A_eq_bezier_shank_constAtZerosL;A_eq_bezier_shank_zerodsLatzerosL;];
A_eq_pelvis = [A_eq_C0_constraint;A_eq_C1_constraint;A_eq_bezier_shank_constAtZerosL;A_eq_bezier_shank_constdsL];


b_eq_foot = [b_eq_C0_constraint;b_eq_C1_constraint;b_eq_shank_constAtZerosL;b_eq_shank_zerodsLatzerosL;];
b_eq_shank = [b_eq_C0_constraint;b_eq_C1_constraint;b_eq_shank_constAtZerosL;b_eq_shank_zerodsLatzerosL;];
b_eq_thigh = [b_eq_C0_constraint;b_eq_C1_constraint;b_eq_shank_constAtZerosL;b_eq_shank_zerodsLatzerosL;];
b_eq_pelvis = [b_eq_C0_constraint;b_eq_C1_constraint;b_eq_shank_constAtZerosL;b_eq_shank_constdsL];

best_fit_params_footAngle = lsqlin(A_dataport_mat_master,b_dataport_footAngle_master,[],[],A_eq_foot, b_eq_foot);
best_fit_params_shankAngle = lsqlin(A_dataport_mat_master,b_dataport_shankAngle_master,[],[],A_eq_shank, b_eq_shank);
best_fit_params_thighAngle = lsqlin(A_dataport_mat_master,b_dataport_thighAngle_master,[],[],A_eq_thigh, b_eq_thigh);
best_fit_params_pelvisAngle = lsqlin(A_dataport_mat_master,b_dataport_pelvisAngle_master,[],[],A_eq_pelvis, b_eq_pelvis);

phase = linspace(0,1,200)';
% incline = 0:0.1:1;
stepLength = 0:0.1:2;
% stepLength_k = 1;


[X,Y] = meshgrid(phase,stepLength);

best_fit_footAngle_mat_311 = zeros(size(X));
best_fit_shankAngle_mat_311 = zeros(size(X));
best_fit_thighAngle_mat_311 = zeros(size(X));
best_fit_pelvisAngle_mat_311 = zeros(size(X));


best_fit_footAngle_mat_312 = zeros(size(X));
best_fit_shankAngle_mat_312 = zeros(size(X));
best_fit_thighAngle_mat_312 = zeros(size(X));
best_fit_pelvisAngle_mat_312 = zeros(size(X));

best_fit_footAngle_mat_313 = zeros(size(X));
best_fit_shankAngle_mat_313 = zeros(size(X));
best_fit_thighAngle_mat_313 = zeros(size(X));
best_fit_pelvisAngle_mat_313 = zeros(size(X));

for i = 1:length(phase)
    
    for j = 1:length(stepLength)
%         i
%         j
        phase_i = X(j,i);
        stepLength_j = Y(j,i);
        best_fit_footAngle_mat_311(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_j,10,best_fit_params_footAngle, phaseDelins,numFuncs);
        best_fit_shankAngle_mat_311(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_j,10,best_fit_params_shankAngle, phaseDelins,numFuncs);
        best_fit_thighAngle_mat_311(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_j,10,best_fit_params_thighAngle, phaseDelins,numFuncs);
        best_fit_pelvisAngle_mat_311(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_j,10,best_fit_params_pelvisAngle, phaseDelins,numFuncs);

        
        best_fit_footAngle_mat_312(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_j,0,best_fit_params_footAngle, phaseDelins,numFuncs);
        best_fit_shankAngle_mat_312(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_j,0,best_fit_params_shankAngle, phaseDelins,numFuncs);
        best_fit_thighAngle_mat_312(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_j,10,best_fit_params_thighAngle, phaseDelins,numFuncs);
        best_fit_pelvisAngle_mat_312(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_j,10,best_fit_params_pelvisAngle, phaseDelins,numFuncs);

        
        best_fit_footAngle_mat_313(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_j,-10,best_fit_params_footAngle, phaseDelins,numFuncs);
        best_fit_shankAngle_mat_313(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_j,-10,best_fit_params_shankAngle, phaseDelins,numFuncs);
        best_fit_thighAngle_mat_313(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_j,10,best_fit_params_thighAngle, phaseDelins,numFuncs);
        best_fit_pelvisAngle_mat_313(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_j,10,best_fit_params_pelvisAngle, phaseDelins,numFuncs);


    end
    
    
end

figure(2)
subplot(1,3,1)
surf(phase, stepLength,best_fit_footAngle_mat_311)
subplot(1,3,2)
surf(phase, stepLength,best_fit_footAngle_mat_312)
subplot(1,3,3)
surf(phase, stepLength,best_fit_footAngle_mat_313)


figure(4)
subplot(1,3,1)
surf(phase, stepLength,best_fit_shankAngle_mat_311)
subplot(1,3,2)
surf(phase, stepLength,best_fit_shankAngle_mat_312)
subplot(1,3,3)
surf(phase, stepLength,best_fit_shankAngle_mat_313)



figure(39)
subplot(1,3,1)
surf(phase, stepLength,best_fit_thighAngle_mat_311)
subplot(1,3,2)
surf(phase, stepLength,best_fit_thighAngle_mat_312)
subplot(1,3,3)
surf(phase, stepLength,best_fit_thighAngle_mat_313)



figure(40)
subplot(1,3,1)
surf(phase, stepLength,best_fit_pelvisAngle_mat_311)
subplot(1,3,2)
surf(phase, stepLength,best_fit_pelvisAngle_mat_312)
subplot(1,3,3)
surf(phase, stepLength,best_fit_pelvisAngle_mat_313)


%% plot against incline

phase = linspace(0,1,200)';
incline = -10:1:10;
stepLength_k = 1;


[X,Y] = meshgrid(phase,incline);

best_fit_footAngle_mat_incline = zeros(size(X));
best_fit_shankAngle_mat_incline = zeros(size(X));
best_fit_thighAngle_mat_incline = zeros(size(X));
best_fit_pelvisAngle_mat_incline = zeros(size(X));


for i = 1:length(phase)
    
    for j = 1:length(incline)
%         i
%         j
        phase_i = X(j,i);
        incline_j = Y(j,i);
        best_fit_footAngle_mat_incline(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_k,incline_j,best_fit_params_footAngle, phaseDelins,numFuncs);
        best_fit_shankAngle_mat_incline(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_k,incline_j,best_fit_params_shankAngle, phaseDelins,numFuncs);
        best_fit_thighAngle_mat_incline(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_k,incline_j,best_fit_params_thighAngle, phaseDelins,numFuncs);
        best_fit_pelvisAngle_mat_incline(j,i) = returnPiecewiseBezier3D(phase_i,stepLength_k,incline_j,best_fit_params_pelvisAngle, phaseDelins,numFuncs);


    end
    
    
end



figure(5)
surf(phase, incline,best_fit_footAngle_mat_incline)
title('Foot Angle')

xlabel('percent gait')
ylabel('Incline, scaled (deg)')
zlabel('degrees'); 
view(60,60)

figure(6)
surf(phase, incline,best_fit_shankAngle_mat_incline)
title('Shank Angle')

xlabel('percent gait')
ylabel('Incline, scaled (deg)')
zlabel('degrees'); 
view(60,60)

figure(7)
surf(phase, incline,best_fit_thighAngle_mat_incline)
title('Thigh Angle')

xlabel('percent gait')
ylabel('Incline, scaled (deg)')
zlabel('degrees'); 
view(60,60)

figure(8)
surf(phase, incline,best_fit_pelvisAngle_mat_incline)
title('Pelvis Angle')

xlabel('percent gait')
ylabel('Incline, scaled (deg)')
zlabel('degrees'); 
view(60,60)


%% save variables


save('gaitModel','A_dataport_mat_master','b_dataport_footAngle_master',...
    'b_dataport_shankAngle_master','best_fit_params_footAngle',...
    'best_fit_params_shankAngle', 'best_fit_params_thighAngle', 'best_fit_params_pelvisAngle')
M = [best_fit_params_footAngle';best_fit_params_shankAngle';...
    best_fit_params_thighAngle';best_fit_params_pelvisAngle'];

writematrix(M,'gaitModel.csv')
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
