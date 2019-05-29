function [tracker, location] = pf_peak_hist_update(tracker, I, varargin)
    im = I;
    
    N = length(tracker.particles);
    Q = tracker.weights / sum(tracker.weights);
    R = cumsum(Q);
    [~,I] = histc(rand(1, N), R); 
    particles = tracker.particles(:, I + 1);
    
    weights = zeros(1, length(Q));
    peak_temp = zeros(1, length(Q));
    hel = zeros(1, length(Q));
    % iterate over each particle
    for particle_n = 1:N
        particle = particles(:, particle_n);
        particles(:, particle_n) = mvnrnd(particle, tracker.model.Q, 1);
        
        % filter      
        patch = get_patch(im, [particles(1, particle_n) particles(2, particle_n)], 1, tracker.sz);
        hogmap = get_feature_map(patch);
        zf = fft2(bsxfun(@times, hogmap, tracker.cos_window));
        kzf = sum(zf .* conj(tracker.xf),3) / numel(zf);
        R = real(ifft2(tracker.alphaf .* kzf));
        
        %find peak
        peak = max(R(:));
        peak_temp(particle_n) = peak;
        weights(particle_n) = peak;
        [dy,dx] = find(R == peak, 1);
        if dy > tracker.sz(2) / 2 
            dy = dy - tracker.sz(2);
        end
        dy = dy - 1;
        if dx > tracker.sz(1) / 2
            dx = dx - tracker.sz(1);
        end
        dx = dx - 1; 
        
        % save the peak location
        particles(1, particle_n) = particles(1, particle_n) + dx;
        particles(2, particle_n) = particles(2, particle_n) + dy;
        
        % hist
        patch = get_patch(im, [particles(1, particle_n) particles(2, particle_n)], 1, tracker.bb_sz);
        hist = extract_histogram(patch, tracker.bins, tracker.kernel);
        hist = hist ./ sum(hist(:));
        hel(particle_n) = sqrt(0.5 * sum((sqrt(hist(:)) - sqrt(tracker.hist(:))) .^2));
    end
    
    %{
    imshow(im);
    hold on;
    plot(particles(1,:), particles(2,:), '*');
    rectangle('Position', [tracker.centre - tracker.sz / 2, tracker.sz]);
    hold off;
    drawnow;
    %}
    
    
    % change distance to probability
    stddev = std(hel);
    
    weights = exp(-((hel .^ 2) / (2 * stddev ^ 2)));
    if isnan(weights)
       weights = ones(1, tracker.particle_num); 
    end
    
    
    
    
    tracker.particles = particles;
    
    weights = weights / sum(weights);
    if not(isnan(weights))
        x = round(sum(weights .* particles(1,:)));
        y = round(sum(weights .* particles(2,:)));
        tracker.centre = [x, y];
        tracker.weights = weights;
    end 
    patch = get_patch(im,tracker.centre,1,tracker.sz);
    hogmap = get_feature_map(patch);
    
    % calculate the translation filter update
    xf = bsxfun(@times, hogmap, tracker.cos_window);
    xf = fft2(xf);
    kf = sum(xf .* conj(xf), 3) / numel(xf);
    alphaf = tracker.yf ./ (kf + eps);   %equation for fast training
    tracker.alphaf = (1 - tracker.alpha) * tracker.alphaf + tracker.alpha * alphaf;
    tracker.xf = (1 - tracker.alpha) * tracker.xf + tracker.alpha * xf;
    
    patch = get_patch(im, tracker.centre, 1, tracker.bb_sz);
    hist = extract_histogram(patch, tracker.bins, tracker.kernel);
    hist = hist ./ sum(hist(:));
    tracker.hist = (1 - tracker.alpha) * tracker.hist + tracker.alpha * hist;
    
    location = [tracker.centre - tracker.bb_sz / 2, tracker.bb_sz];
    
    
    hold on;
    scatter(particles(1,:), particles(2,:), 200*weights, 'filled');
    rectangle('Position', [tracker.centre - tracker.bb_sz / 2, tracker.bb_sz], 'LineWidth',2, 'EdgeColor','r');
    hold off;
    %drawnow;
    
    
    
end
