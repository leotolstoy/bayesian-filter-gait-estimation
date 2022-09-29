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

% sub={'AB01'};

% The command 'fieldnames' also returns the name of all fields within a
% struct as a cell array

trials = {'s1i10','s1i5','s1i0','s1d5','s1d10',...
            's1x2i10','s1x2i5','s1x2i0','s1x2d5','s1x2d10',...
            's0x8i10','s0x8i5','s0x8i0','s0x8d5','s0x8d10'}
% trials = {'s1i0'}

% trials = fieldnames(Gaitcycle.AB01);
% The command 'setdiff' is useful for removing cells of a given name
trials = setdiff(trials,'subjectdetails');


leg={'left'};
percent_gait = linspace(0,1,150);






stepMin = 0.9;
stepMax = 1.5;

inclineMin = -10;
inclineMax = 10;

phaseDelins = [0.1,0.5,0.65,1]


% time_offset = 0;

% time_vec_master = [];
% shankAngle_vec_master = [];
% footAngle_vec_master = [];
% 
% shankAngleVel_vec_master = [];
% footAngleVel_vec_master = [];
% 
% HSDetected_vec_master = [];
% 
% phase_vec_master = [];
% phase_dot_vec_master = [];
% stepLength_descaled_vec_master = [];
% incline_vec_master = [];

DOWNSAMPLE = false
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
    

    for j = 1:length(trials) %loop through all trials in 'trial'
        
        time_offset = 0;
        time_vec_master = [];
        shankAngle_vec_master = [];
        footAngle_vec_master = [];
        thighAngle_vec_master = [];
        pelvisAngle_vec_master = [];

        shankAngleVel_vec_master = [];
        footAngleVel_vec_master = [];
        thighAngleVel_vec_master = [];
        pelvisAngleVel_vec_master = [];

        HSDetected_vec_master = [];

        phase_vec_master = [];
        phase_dot_vec_master = [];
        stepLength_descaled_vec_master = [];
        incline_vec_master = [];
    
        % store the kinematic data in a temporary variable
%         sub{i}
        trial = trials{j};
        foot_angle_data = Gaitcycle.(sub{i}).(trials{j}).kinematics.jointangles.(leg{1}).('foot').x;
        ankle_angle_data = Gaitcycle.(sub{i}).(trials{j}).kinematics.jointangles.(leg{1}).('ankle').x;
        knee_angle_data = Gaitcycle.(sub{i}).(trials{j}).kinematics.jointangles.(leg{1}).('knee').x;
        hip_angle_data = Gaitcycle.(sub{i}).(trials{j}).kinematics.jointangles.(leg{1}).('hip').x;
        pelvis_angle_data = Gaitcycle.(sub{i}).(trials{j}).kinematics.jointangles.(leg{1}).('pelvis').x;
        foot_angle_data = -(foot_angle_data + footAngleZero);
        
        time_data = Gaitcycle.(sub{i}).(trials{j}).cycles.(leg{1}).time;
        
        foot_angle_data(:,Gaitcycle.(sub{i}).(trials{j}).stepsout.(leg{1})) = [];
        ankle_angle_data(:,Gaitcycle.(sub{i}).(trials{j}).stepsout.(leg{1})) = [];
        knee_angle_data(:,Gaitcycle.(sub{i}).(trials{j}).stepsout.(leg{1})) = [];
        hip_angle_data(:,Gaitcycle.(sub{i}).(trials{j}).stepsout.(leg{1})) = [];
        pelvis_angle_data(:,Gaitcycle.(sub{i}).(trials{j}).stepsout.(leg{1})) = [];
        
        time_data(:,Gaitcycle.(sub{i}).(trials{j}).stepsout.(leg{1})) = [];
        
        shank_angle_data = foot_angle_data - ankle_angle_data;
        thigh_angle_data = pelvis_angle_data + hip_angle_data;
        
        [treadmillSpeed, treadmillIncline] = returnSpeedIncline(trial);
        incline = treadmillIncline;

        [rows,cols] = size(foot_angle_data);
                
        for k = 1:cols
            
            if DOWNSAMPLE
                if k == 1
                    idxs = 1:3:150;
                else
                    idxs = 2:3:150;

                end
                
            else
                if k == 1
                    idxs = 1:150;
                else
                    idxs = 2:150;

                end
                
            end
            foot_angle_data_col = foot_angle_data(idxs,k);
            shank_angle_data_col = shank_angle_data(idxs,k);
            thigh_angle_data_col = thigh_angle_data(idxs,k);
            pelvis_angle_data_col = pelvis_angle_data(idxs,k);
            
            time_data_col = time_data(idxs,k);
%             time_data_col(1)
%             time_data_col(end)
%             time_offset
%             pause
                        
            time_vec_master = [time_vec_master;time_data_col + time_offset];
            
            footAngle_vec_master = [footAngle_vec_master;foot_angle_data_col];
            shankAngle_vec_master = [shankAngle_vec_master;shank_angle_data_col];
            thighAngle_vec_master = [thighAngle_vec_master;thigh_angle_data_col];
            pelvisAngle_vec_master = [pelvisAngle_vec_master;pelvis_angle_data_col];
            
            time_offset = time_offset + time_data_col(end);

            
            
            stepDuration = time_data_col(end) - time_data_col(1);
            phaseDot = 1/stepDuration;
            stepLength = treadmillSpeed*stepDuration;
            
%             stepLength_scaled = (stepLength - stepMin)/(stepMax - stepMin);
            
            phase_col = time_data_col/stepDuration;
            
            phase_vec_master = [phase_vec_master;phase_col];
            phase_dot_vec_master = [phase_dot_vec_master; phaseDot*ones(size(phase_col))];
            stepLength_descaled_vec_master = [stepLength_descaled_vec_master;stepLength*ones(size(phase_col))];
            incline_vec_master = [incline_vec_master;incline*ones(size(phase_col))];
            
            HSDetected_vec = zeros(size(phase_col));
            HSDetected_vec(1) = 1;
            
            HSDetected_vec_master = [HSDetected_vec_master;HSDetected_vec];
            
            
        end
        %     shankAngleVel_vec_master = lowpass(diff(shankAngle_vec_master)./(diff(time_vec_master)),20,100);
    %     footAngleVel_vec_master = lowpass(diff(footAngle_vec_master)./(diff(time_vec_master)),20,100);
        shankAngleVel_vec_master = diff(shankAngle_vec_master)./(diff(time_vec_master));
        footAngleVel_vec_master = diff(footAngle_vec_master)./(diff(time_vec_master));
        thighAngleVel_vec_master = diff(thighAngle_vec_master)./(diff(time_vec_master));
        pelvisAngleVel_vec_master = diff(pelvisAngle_vec_master)./(diff(time_vec_master));
        
        shankAngleVel_vec_master = [shankAngleVel_vec_master(1);shankAngleVel_vec_master ];
        footAngleVel_vec_master = [footAngleVel_vec_master(1);footAngleVel_vec_master ];
        thighAngleVel_vec_master = [thighAngleVel_vec_master(1);thighAngleVel_vec_master ];
        pelvisAngleVel_vec_master = [pelvisAngleVel_vec_master(1);pelvisAngleVel_vec_master ];

        figure(1)
        
        subplot(4,1,1)
        plot(time_vec_master,footAngle_vec_master,'LineWidth',2)
        hold on
        plot(time_vec_master,HSDetected_vec_master*10,'k','LineWidth',2)

        legend('footAngle','HSDetected')
        title(['subject: ', sub{i}])

        subplot(4,1,2)
        plot(time_vec_master,footAngleVel_vec_master,'LineWidth',2)
        legend('footAngleVel')
        subplot(4,1,3)
        plot(time_vec_master,shankAngle_vec_master,'LineWidth',2)
        legend('shankAngle')
        subplot(4,1,4)
        plot(time_vec_master,shankAngleVel_vec_master,'LineWidth',2)
        legend('shankAngleVel')
        xlabel('Time (s)')

        figure(2)
        subplot(4,1,1)
        plot(time_vec_master,phase_vec_master,'LineWidth',2)
        hold on
        plot(time_vec_master,HSDetected_vec_master,'k','LineWidth',2)
        legend('phase','HSDetected')
        subplot(4,1,2)
        plot(time_vec_master,phase_dot_vec_master,'LineWidth',2)
        legend('phase dot')
        subplot(4,1,3)
        plot(time_vec_master,stepLength_descaled_vec_master,'LineWidth',2)
        legend('step length')
        subplot(4,1,4)
        plot(time_vec_master,incline_vec_master,'LineWidth',2)
        legend('incline')
        xlabel('Time (s)')

        %% save files
        if DOWNSAMPLE
            filename = ['dataport_',sub{i},'_',trials{j},'_downsample.csv'];
        else
            filename = ['dataport_',sub{i},'_',trials{j},'.csv'];
        end

        sampling_rate = 1/mean(diff(time_vec_master))

        M = [time_vec_master,...
            footAngle_vec_master,footAngleVel_vec_master,...
            shankAngle_vec_master,shankAngleVel_vec_master,...
            thighAngle_vec_master,thighAngleVel_vec_master,...
            pelvisAngle_vec_master,pelvisAngleVel_vec_master,...
            phase_vec_master, phase_dot_vec_master, stepLength_descaled_vec_master, incline_vec_master,...
            HSDetected_vec_master,...
            foot_angle_mean*ones(size(time_vec_master)),...
            shank_angle_mean*ones(size(time_vec_master)),...
            thigh_angle_mean*ones(size(time_vec_master)),...
            pelvis_angle_mean*ones(size(time_vec_master))];
      
        writematrix(M,filename)
%         pause
        close all

    
      
    end
end











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
