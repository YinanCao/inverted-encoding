%% IEM_localizer_crossval.m
% Inverted encoding model (IEM) on MEG sensor data from the sensory
% localizer, with leave-one-run-out cross-validation.
%
% This follows the method of code_v0/IEM/SL_crossVal_perm.m (MEGframing2022)
% step by step, for one subject, without the permutation test. It runs on
% the slim localizer files in ./localizerData (made by
% extract_localizer_slim.m: -100..500 ms, 100 Hz, single precision).
%
% Needs only base MATLAB plus covdiag_shrink.m (in this folder).
%
% The task in one paragraph
%   On every trial a stimulus appears at one of 3 locations (0, 120, 240 deg;
%   0 = top, counter-clockwise). We assume each MEG sensor's response is a
%   weighted sum of 3 hypothetical "location channels", each tuned to one
%   location. Training: estimate the sensor weights of each channel from
%   the training runs. Testing: invert the weights to turn the held-out
%   run's sensor data back into 3 channel responses per trial, then ask
%   which location those channel responses point to.

clear; clc; close all;

%% 0. Settings -------------------------------------------------------------
subjID   = 1;                         % subject number, as in the file name (..._SUBJ_<n>_...)
dataDir  = fullfile(fileparts(mfilename('fullpath')), 'localizerData');
outDir   = fullfile(fileparts(mfilename('fullpath')), 'IEMresults');

% same values as code_v0/generic_IEM_configs.m
IEMconfig            = [];
IEMconfig.baselineSL = [-0.1, 0];     % baseline window (s)
IEMconfig.power      = 5;             % exponent of the basis functions
IEMconfig.SLoptimal  = [0.14, 0.17];  % window used for training in the main analysis
toiEnd               = 0.45;          % last time point analysed (as in SL_crossVal_perm)

%% 1. Load data ------------------------------------------------------------
f = dir(fullfile(dataDir, sprintf('*_SUBJ_%d_SL_slim.mat', subjID)));
% assert(numel(f) == 1, 'No (unique) localizer file for subject %d in %s', subjID, dataDir);
load(fullfile(dataDir, f.name), 'meg_trl', 'time', 'trlinfo', 'designM');
disp(f.name)

% the slim files store data as single and info as integers; compute in double
meg_trl  = double(meg_trl);           % [sensor x time x trial]
trlinfo  = double(trlinfo);           % [stimulus trigger, run (block) index]
designM  = double(designM);           % [valid, location (1-3), ?, trigger]
timelist = double(time);              % seconds, 0 = stimulus onset

%% 2. Remove flicker (catch) trials ----------------------------------------
stimtrig = trlinfo(:,1);              % stim trigger value
invalid  = ismember(stimtrig, 29:31); % flicker trials (button pressed)
trlinfo(invalid,:)   = [];
meg_trl(:,:,invalid) = [];
megData_SL = meg_trl; clear meg_trl

SL_blockID = trlinfo(:,2);            % run index, used for cross-validation

% look up each trial's design row from its trigger:
% trldesign(:,1) = stimulus location (1 = 0 deg, 2 = 120 deg, 3 = 240 deg)
ntrl = size(trlinfo,1);
trldesign = [];
for trl = ntrl:-1:1
    trldesign(trl,:) = designM(designM(:,end)==trlinfo(trl,1), 2:end);
end

%% 3. Basis functions (the "design matrix" C) ------------------------------
% Each channel's ideal response to a stimulus at location mu is
% |cos(0.5*(theta - mu))|^power, i.e. a half-cosine raised to a power.
theta = [0,120,240]';                 % preferred location of each IEM channel
pref  = [0,120,240];                  % location (deg) of stimulus 1, 2, 3
C = [];                               % [channel x trial]
for trl = ntrl:-1:1
    mu = pref(trldesign(trl,1));
    C(:,trl) = abs(cos(0.5*(theta-mu)/180*pi).^IEMconfig.power);
end

%% 4. Baseline correction (per trial, per sensor) --------------------------
baselineSL = IEMconfig.baselineSL;
base_t_SL  = timelist<=baselineSL(2) & timelist>=baselineSL(1);
BL = mean(megData_SL(:,base_t_SL,:),2,'omitnan');
megData_SL_bl = megData_SL - BL;      % [sensor x time x trial]

% time points of interest
rmv_idx = timelist<baselineSL(1) | timelist>toiEnd;
megData_SL_bl(:,rmv_idx,:) = [];
timelist(rmv_idx) = [];

%% 5. Drop rejected (NaN) trials -------------------------------------------
tmp      = squeeze(mean(megData_SL_bl,2,'omitnan'));
noiseTrl = isnan(tmp(1,:));           % rejected trials are NaN on every sensor

C_noNaN          = C;
SL_blockID_noNaN = SL_blockID;
C_noNaN(:,noiseTrl)         = [];
SL_blockID_noNaN(noiseTrl)  = [];
megData_SL_bl(:,:,noiseTrl) = [];
% NOTE: SL_crossVal_perm.m does not remove these trials from trldesign,
% so its accuracy step would pair trials with the wrong labels whenever a
% subject has rejected trials. Removing them here keeps labels aligned.
% (Nothing changes for subjects without rejected trials.)
trldesign(noiseTrl,:) = [];
fprintf('%d trials, %d removed as NaN\n', size(megData_SL_bl,3), sum(noiseTrl));

%% 6. Demeaning happens inside the cross-validation loop ------------------
% Within each fold, the design matrix and the MEG data are centred using the
% mean of the TRAINING trials only, and that same training mean is
% subtracted from the test trials. The held-out run therefore never
% influences anything the model is fitted on.
% (SL_crossVal_perm.m instead demeaned across ALL trials before the split:
%   C_noNaN = C_noNaN - mean(C_noNaN,2);
%   megData_SL_bl = megData_SL_bl - mean(megData_SL_bl,3);
%  which lets test trials leak into the centring.)

%% 7. Cross-validated IEM: train and test at each time point ---------------
nt    = length(timelist);
runID = unique(SL_blockID_noNaN);
nrun  = length(runID);
nch   = size(C_noNaN,1);
nTrlKeep = size(megData_SL_bl,3);

IEM_crossval = nan(nch, nt, nTrlKeep);   % channel responses [channel x time x trial]
tic
for t_train = nt:-1:1
    for whichrun = 1:nrun
        trl_test  = SL_blockID_noNaN==runID(whichrun);   % leave this run out
        trl_train = ~trl_test;
        Dtrain  = squeeze(megData_SL_bl(:,t_train,trl_train));  % [sensor x trial]
        Dtest   = squeeze(megData_SL_bl(:,t_train,trl_test));
        C_train = C_noNaN(:,trl_train);                         % [channel x trial]

        % centre with the training mean only (applied to train AND test)
        mu      = mean(Dtrain,2);                               % [sensor x 1]
        Dtrain  = Dtrain - mu;
        Dtest   = Dtest  - mu;
        C_train = C_train - mean(C_train,2);

        V = [];
        for ch = nch:-1:1                 % estimate each channel separately
            % (a) encoding weights of this channel on every sensor (least squares)
            w     = Dtrain*C_train(ch,:)'*pinv(C_train(ch,:)*C_train(ch,:)');
            % (b) noise covariance of the residuals (training trials only),
            %     shrunk towards its diagonal
            err   = Dtrain - w*C_train(ch,:);
            sigma = covdiag_shrink(err');
            % (c) noise-whitened inverse weights (a spatial filter for this channel)
            v     = (sigma^-1)*w/(w'*(sigma^-1)*w);
            V(:,ch) = v;
        end
        % (d) apply the filters to the left-out run: channel responses
        IEM_crossval(:,t_train,trl_test) = V'*Dtest;   % = mtimesx(V,'t',Dtest)
    end
    fprintf('time %d/%d (%.0f ms)\n', nt-t_train+1, nt, 1000*timelist(t_train));
end
toc

%% 8. From channel responses to a location estimate ------------------------
% Treat the 3 channel responses as a vector sum on the circle: each channel
% points towards its preferred location (+90 deg so that 0 deg is up).
y      = IEM_crossval;                                    % [channel x time x trial]
neur_x = sum(y .* cos((theta+90)/180*pi), 1);             % [1 x time x trial]
neur_y = sum(y .* sin((theta+90)/180*pi), 1);
xy_wA  = cat(1, neur_x, neur_y);                          % [2 x time x trial]

stim_ref = [0,120,240];
xy_ref   = [cos((stim_ref+90)*pi/180); sin((stim_ref+90)*pi/180)];   % unit vectors

l = squeeze(sqrt(sum(xy_wA.^2,1)));                       % vector length [time x trial]
cosSim = [];
for loc = 3:-1:1
    % projection of the population vector onto location loc
    aaa = squeeze(xy_wA(1,:,:)*xy_ref(1,loc) + xy_wA(2,:,:)*xy_ref(2,loc));  % [time x trial]
    cosSim(loc,:,:,1) = aaa./l;       % cosine similarity
    cosSim(loc,:,:,2) = aaa;          % projection (length-weighted)
end
cosSim = cat(4, cosSim, y);           % 3rd read-out: the raw channel responses
% cosSim: [location x time x trial x read-out]

%% 9. Decoding accuracy ----------------------------------------------------
% A trial counts as correct when the value for the shown location is larger
% than for both other locations (SL_crossVal_perm uses prctile(...,100,1),
% which is the maximum). Chance = 1/3.
ntrl = size(cosSim,3);
tmp  = [];
for trl = ntrl:-1:1
    stimloc = trldesign(trl,1);
    emptloc = setdiff(1:3, stimloc);
    acc = cosSim(stimloc,:,trl,:) > max(cosSim(emptloc,:,trl,:), [], 1);
    tmp(trl,:,:) = squeeze(acc);      % [trial x time x read-out]
end
% stimulus type: top, left, right, all
type = [trldesign(:,1)==1, trldesign(:,1)==2, trldesign(:,1)==3, trldesign(:,1)>0];
decoding_acc = [];
for c = size(type,2):-1:1
    decoding_acc(:,:,c) = squeeze(mean(tmp(type(:,c),:,:),1,'omitnan'));
end
dim_info = {'time','read-out: cosSim,proj,raw','stim type: top,left,right,all'};

%% 10. Save ----------------------------------------------------------------
if ~exist(outDir,'dir'); mkdir(outDir); end
fOut = fullfile(outDir, sprintf('IEM_crossval_SUBJ_%d.mat', subjID));
save(fOut, 'decoding_acc', 'IEMconfig', 'timelist', 'dim_info', 'IEM_crossval', 'trldesign');
disp(['saved ', fOut])

%% 11. Plots ---------------------------------------------------------------
locNames = {'top (0°)','left (120°)','right (240°)','all'};
ms = 1000*timelist;

% (a) decoding accuracy over time (cosine-similarity read-out)
figure('Color','w','Position',[100 100 900 380]);
subplot(1,2,1); hold on
plot(ms, decoding_acc(:,1,4), 'k', 'LineWidth', 2)
yline(1/3, '--', 'chance'); xline(0, ':');
xregion = IEMconfig.SLoptimal*1000;
patch([xregion fliplr(xregion)], [0 0 1 1], [0.8 0.8 0.8], ...
      'EdgeColor','none','FaceAlpha',0.3);
xlabel('time (ms)'); ylabel('decoding accuracy'); ylim([0.2 1]);
title(sprintf('Subject %d: all locations', subjID)); box off

subplot(1,2,2); hold on
plot(ms, squeeze(decoding_acc(:,1,1:3)), 'LineWidth', 1.5)
yline(1/3, '--'); xline(0, ':');
legend(locNames(1:3), 'Location','northwest'); legend boxoff
xlabel('time (ms)'); ylabel('decoding accuracy'); ylim([0.2 1]);
title('per stimulus location'); box off

% (b) mean channel responses in the 140-170 ms window, per stimulus location
win = timelist>=IEMconfig.SLoptimal(1) & timelist<=IEMconfig.SLoptimal(2);
chanResp = squeeze(mean(IEM_crossval(:,win,:),2));       % [channel x trial]
figure('Color','w','Position',[100 550 900 300]);
for loc = 1:3
    subplot(1,3,loc)
    bar(theta, mean(chanResp(:, trldesign(:,1)==loc), 2))
    xlabel('channel preference (deg)'); ylabel('response (a.u.)');
    title(['stimulus ', locNames{loc}]); box off
end
sgtitle('Reconstructed channel responses, 140-170 ms')

% Note: with 3 equally spaced channels, the three read-out METHODS (cosine
% similarity, projection, raw channel response; 2nd dimension of
% decoding_acc) pick the same winning location on every trial, so they give
% identical accuracy. They differ only when more channels are used (change
% theta and the basis set). This is not about the per-location curves
% (top, left, right; 3rd dimension): those can still differ from each other.
