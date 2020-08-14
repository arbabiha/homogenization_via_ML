function u_tiled = tile_patches(u)

% we put back usim snapshot into 2d format
shape = size(u);

nx = shape(1)*shape(3);
ny = shape(2)*shape(4);

u_tiled = zeros(ny,nx);



nxp = shape(1); % no. of points within patches
nyp = shape(2);

npx = shape(3); % no. of patches
npy = shape(4);


for i=1:npx
    for j=1:npy
        u_tiled( (j-1)*nyp+1:j*nyp, (i-1)*nxp+1:i*nxp ) = u(:,:,i,j)';
    end
end