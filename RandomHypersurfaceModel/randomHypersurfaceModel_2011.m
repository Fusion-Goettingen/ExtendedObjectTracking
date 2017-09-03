% Implementation of Star-Convex RHMs using Fourier Coefficients and the UKF according to the paper 
%
% M. Baum and U. D. Hanebeck, "Shape tracking of extended objects and group targets with star-convex RHMs," 
% 14th International Conference on Information Fusion, Chicago, IL, 2011, pp. 1-8.
% URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5977661&isnumber=5977431
%
% Source code written by Robin Sandkuehler & Marcus Baum 

function randomHypersurfaceModel_2011(numberOfMeasurement)
if nargin ==0
numberOfMeasurement= 100;
end
% Number of Fourier coefficients
nr_Fourier_coeff = 11;

% State describtion prior [b0--bn, x, y]
x = zeros(nr_Fourier_coeff + 2, 1);
x(1) = 1.5;

% State covariance prior
C_x = diag([ones(1, nr_Fourier_coeff).*0.02, 0.3, 0.3]);

% Measurement noise
measurementNoise = diag([0.2, 0.2].^2);

% Scale properties
scale.mean = 0.7;
scale.variance = 0.08;

% Angular resolution for plotting
phi_vec = [0:0.01:2*pi];

% Object size
a = 3;      % -- width of the horizontal rectangle
b = 0.5;    % | height of the horizontal rectangle
c = 2;      % | height of the vertical rectangle
d = 0.5;    % -- width of the vertical rectangle

sizeObject = [a b c d];

% Object shape bounds
objectBounds = [[-d, -c];[d, -c];[d, -b];[a, -b];[a, b];[d, b];[d, c];
    [-d, c];[-d, b];[-a, b];[-a, -b];[-d, -b]]' ./ 2;


% Main

% Plot
h_object = fill(objectBounds(1, :), objectBounds(2, :), [.7 .7 .7]);
hold on
xlim([-3 3]);
ylim([-3 3]);
axis equal
xlabel('x-Axis')
ylabel('y-Axis')
title('Random Hypersurface Model Simulation')



for j = 1 : numberOfMeasurement
    
    % Get new measurement
    newMeasurement = getNewMeasurement(sizeObject, measurementNoise);
    
    % Filter step
    [x, C_x] = UKF_FilterStep(x, C_x, newMeasurement, [scale.mean; [0 0]'], ...
        blkdiag(scale.variance, measurementNoise), @f_meas_pseudo_squared, nr_Fourier_coeff);
    
    % Plot
    shape = calcShape(phi_vec, x, nr_Fourier_coeff);
    
    h_measure = plot(newMeasurement(1), newMeasurement(2), '+');
    h_shape =  plot(shape(1, :), shape(2, :), 'g-', 'linewidth', 2);
    legend([h_object, h_measure, h_shape],'Target', 'Measurement', 'Estimated shape')
    drawnow;
    
    if j ~= numberOfMeasurement
        delete(h_shape)
    end
    
end

hold off
end



function measurement =  getNewMeasurement( sizeObject, measurementNoise )

a = sizeObject(1); % -- width of the horizontal rectangle
b = sizeObject(2); % | height of the horizontal rectangle
c = sizeObject(3); % | height of the vertical rectangle
d = sizeObject(4); % -- width of the vertical rectangle

measurementsourceNotValid = 1;

while measurementsourceNotValid
    
    %Measurementsoure
    x = -a/2 + a.*rand(1, 1);
    y = -c/2 + c.*rand(1, 1);
    
    if y > b/2 && x < -d/2 || y > b/2 && x > d/2 ||...
       y < -b/2 && x < -d/2 || y < -b/2 && x > d/2
        
        x = -a/2 + a.*rand(1, 1);
        y = -c/2 + c.*rand(1, 1);
        
    else
        measurementsourceNotValid = 0;
    end
end

% Add zero-mean Gaussian noise to the measurement sources
measurement = [x; y] + (randn(1, 2) * chol(measurementNoise))';
end

function pseudoMeasurement = f_meas_pseudo_squared(x, noise, y, nr_Fourier_coeff)

numberOfSigmaPoints = size(x, 2);
pseudoMeasurement = zeros(1, numberOfSigmaPoints);

for j = 1 : numberOfSigmaPoints
    
    s = noise(1, j);
    v = noise(2:3, j);
    b = x(1:nr_Fourier_coeff, j);
    m = x(nr_Fourier_coeff + 1:nr_Fourier_coeff + 2, j);
    
    theta = atan2(y(2) - m(2), y(1) - m(1))+2*pi;

    R = calcFourierCoeff(theta, nr_Fourier_coeff);
    
    e = [cos(theta); sin(theta)];
    
    pseudoMeasurement(j) = (norm( m - y ))^2 - (s^2 *(R * b).^2 + 2 * s * R * b * e' * v + norm(v)^2);
end
end




function fourie_coff = calcFourierCoeff(theta, nr_Fourier_coeff)

fourie_coff(1) = 0.5;

index = 1;
for i = 2 : 2 : (nr_Fourier_coeff - 1)
    fourie_coff(i : i + 1) = [cos(index * theta) sin(index * theta)];
    index = index + 1;
end


end


function shape = calcShape(phi_vec, x, nr_Fourier_coeff)

shape = zeros(2, length(phi_vec));

for i = 1 : length(phi_vec)
    phi = phi_vec(i);
    R = calcFourierCoeff(phi, nr_Fourier_coeff);
    e = [cos(phi) sin(phi)]';
    shape(:, i) = R * x(1:end - 2) * e + x(end - 1:end);
end

end

function [x_e, C_e] = UKF_FilterStep(x, C, measurement, measurementNoiseMean, ...
    measurementNoiseCovariance, measurementFunctionHandle, numberOfFourierCoef)

alpha = 1;
beta = 0;
kappa = 0;

% Calculate Sigma Points

%Stack state and noise mean
x_ukf = [x; measurementNoiseMean];


%Stack state and noise Covariance
C_ukf = blkdiag(C, measurementNoiseCovariance);

n = size(x_ukf, 1);
n_state = size(x, 1);

lamda = alpha^2 * (n + kappa) - n;


% Calculate Weights Mean
WM(1) = lamda / (n + lamda);
WM(2 : 2 * n + 1) = 1 / (2 * (n + lamda));
% Calculate Weights Covariance
WC(1) = (lamda / (n + lamda)) + (1 - alpha^2 + beta);
WC(2 : 2 * n + 1) = 1 / (2 * (n + lamda));

%Calculate Sigma Points
A = sqrt(n + lamda) * chol(C_ukf)';

xSigma = [zeros(size(x_ukf)) -A A];
xSigma = xSigma + repmat(x_ukf, 1, size(xSigma, 2));

% Filterstep
z = 0;
C_yy = 0;
C_xy = 0;

zSigmaPredict = feval(measurementFunctionHandle, xSigma(1:n_state,:), xSigma(n_state + 1:n, :), measurement, numberOfFourierCoef );

for i = 1 : size(zSigmaPredict, 2);
    z = z + ( zSigmaPredict(:,i) * WM(i) );
end


for i = 1 : size(zSigmaPredict, 2)
    C_yy = C_yy + WC(i) * ( (zSigmaPredict(:,i) - z ) * ( zSigmaPredict(:,i) - z )') ;
    C_xy = C_xy + WC(i) * ( (xSigma(1:size(x, 1),i) - x ) * ( zSigmaPredict(:,i) - z )');
end

K = C_xy / C_yy;
x_e = x + K * (zeros(size(z)) - z);
C_e = C - K * (C_yy) * K';

end