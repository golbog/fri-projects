function tracker = pf_peak_hist_initialize(I, region, varargin)
    % CENTRE AND PATCH SIZE
    im = I;
    [height, width] = size(im);

    % If the provided region is a polygon ...
    if numel(region) > 4
        x1 = round(min(region(1:2:end)));
        x2 = round(max(region(1:2:end)));
        y1 = round(min(region(2:2:end)));
        y2 = round(max(region(2:2:end)));
        region = round([x1, y1, x2 - x1, y2 - y1]);
    else
        region = round([round(region(1)), round(region(2)), ... 
            round(region(1) + region(3)) - round(region(1)), ...
            round(region(2) + region(4)) - round(region(2))]);
    end

    x1 = max(1, region(1));
    y1 = max(1, region(2));
    x2 = min(width-2, region(1) + region(3) - 1);
    y2 = min(height-2, region(2) + region(4) - 1);
    
    w = floor(x2 - x1 + 1);
    h = floor(y2 - y1 + 1);
    w = w + (mod(w, 2) == 0);
    h = h + (mod(h, 2) == 0); 
    x = floor((x1 + x2 + 1) / 2);
    y = floor((y1 + y2 + 1) / 2);
    sz = [w, h];
    centre = [x, y]; 
    bb_sz = sz;
    scale = 1;
    sz = floor(sz * scale);
    
    % MOTION MODELS
    q = 1000; 
    RW = struct('A', [1 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0],...
        'C', [1 0 0 0; 0 1 0 0],...
        'Q', [1/3 0 0 0; 0 1/3 0 0; 0 0 0 0;0 0 0 0],...
        'R', [1 0; 0 1],...
        'Xa', []);
    NCV = struct('A', [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1],...
        'C', [1 0 0 0; 0 1 0 0],...
        'Q', [1/3 0 1/2 0; 0 1/3 0 1/2; 1/2 0 1 0; 0 1/2 0 1],...
        'R', [1 0; 0 1],...
        'Xa', []);
    NCA = struct('A', [1 0 1 0 1/2 0; 0 1 0 1 0 1/2; 0 0 1 0 1 0; 0 0 0 1 0 1; ...
        0 0 0 0 1 0; 0 0 0 0 0 1],...
        'C', [1 0 0 0 0  0; 0 1 0 0 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1],...
        'Q', [1/3 0 1/2 0 0 0; 0 1/3 0 1/2 0 0; 1/2 0 1 0 0 0; 0 1/2 0 1 0 0;...
        0 0 0 0 0 0; 0 0 0 0 0 0],...
        'R', [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1],...
        'Xa', [0 0]);
    model = NCV;    
    model.Q = q * model.Q;  
    
    % PARTICLES
    noise = 100;
    n_particles = 50;
    
    particles = [mvnrnd([x y], [noise 0;0 noise], n_particles)'; ...
             zeros(size(model.A, 1) - 2, n_particles)]; 
    weights = ones(1, n_particles);
    
    %{
    imshow(im);
    hold on;
    plot(particles(1,:), particles(2,:), '*');
    rectangle('Position', [centre - sz / 2, sz]);
    hold off;fds
    draw;
    %}
    
    % CORRELATION 
    eps = 1e-2;
    alpha = 0.2;
    output_simga_ratio = 0.01;
    
	output_sigma = sqrt(prod(bb_sz)) * output_simga_ratio;
	yf = fft2(gaussian_shaped_labels(output_sigma, sz));
    yf = yf.';
    
    cos_window = create_cos_window(sz);
    patch = get_patch(im,centre,1,sz);
    hogmap = get_feature_map(patch);
    
    xf = bsxfun(@times, hogmap, cos_window);
    xf = fft2(xf);
    kf = sum(xf .* conj(xf), 3) / numel(xf);
    alphaf = yf ./ (kf + eps); 
    
    % HISTOGRAM
    patch = get_patch(im, centre, 1, bb_sz);
    kernel = create_epanechnik_kernel(w, h, 1); 
    bins = 16;
    hist = extract_histogram(patch, bins, kernel);
    hist = hist ./ sum(hist(:));
    
    % RETURN
    tracker = struct('sz', sz, 'bb_sz', bb_sz, 'centre', centre,...
        'model', model, 'particles', particles, 'weights', weights, 'particle_num', n_particles, ...
        'xf', xf, 'yf', yf, 'alphaf', alphaf, 'cos_window', cos_window, ...
        'eps', eps, 'alpha', alpha,...
        'hist', hist, 'bins', bins, 'kernel', kernel);

end