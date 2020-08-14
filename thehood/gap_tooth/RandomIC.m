function V = RandomIC(x,y)
% generating random fields for x,y
% x and y are two matrices of the same size.
% the IC is a*sin(k1*x+p1)sin(k2*y+p2)
% k1,k2 are random integers, p1 and p2 are 0 or pi
% a is randomly chosen between -1 and 1
rng shuffle
% randi(1000)

n_modes = 10;

max_wave_number = 5;

k1 = randi(max_wave_number,n_modes);
k2 = randi(max_wave_number,n_modes);

p1 = pi * ( randi(2,n_modes)-1);
p2 = pi * ( randi(2,n_modes)-1);

a = 2*rand(n_modes)-1;


V = zeros(size(x));

for j=1:n_modes
    V = V + a(j) .* sin(k1(j)*x + p1(j)) .* sin(k2(j)*y + p2(j));
end

end
    

