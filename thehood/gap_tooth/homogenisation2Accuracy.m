% Patch scheme of heterogeneous diffusion in 2D, and explore
% accuracy as patches get smaller.  Start with full domain
% evals (from case r=1), then smaller patches keeping
% microscale the same. AJR, 16 Apr 2020 -- 17 Jun 2020

clear all
% mPeriod = 3
% a = exp(1*randn(mPeriod));
% b = exp(1*randn(mPeriod));

load('hom2d_data_16x3.mat','C')
cHetr = squeeze(C(1,:,:,:));
a = cHetr(:,:,1);
b = cHetr(:,:,2);
mPeriod=size(a,1);

abfac=mean(1./[a(:);b(:)])  % some arbitrary normalisation
a = a*abfac, b = b*abfac
nPatch = [5 5]
listPeriods = [4 2 1]
baseRatios = 0.25

% % if 1 % can we adapt to cater for this case??
% listPeriods = [1]
% baseRatios = [0.5 0.4]
% nPatch=[10 11]
% a=[18.9097    1.0615    0.6286    2.1132
%    4.4565    0.7212    1.0218    1.6635
%    4.8931    0.8754    1.3100    5.7851
%    1.6224    2.6806    2.3169    1.2433
%    0.4242    0.8817    0.5935    1.3450 ]
% b=[0.4769    0.6302    1.3079    0.5133
%    0.3851   10.3752    3.0678    0.3735
%    2.0989    1.7387    2.6823    1.6331
%    1.1989    4.3839    0.5019    1.0198
%    2.5463    1.2263    0.3323    1.0649 ]
% % end
 
mPeriod=size(a)

global patches
nSmallEvals=30;  gravEvals=[];
for nPeriods=listPeriods
    disp('')
    disp('Patch scheme with the following ratio')
    ratio = nPeriods*baseRatios % ratio=1 is full-domain 
    nSubP = nPeriods*mPeriod+2  % set patch size for the periods
    patches.EdgyInt = 1; % one to use edges for interpolation
    patches.EdgyEns=0; % one for ensemble average
    configPatches2(@heteroDiff2,[-pi pi],nan,nPatch ...
        ,0,ratio,nSubP);

    % replicate diffusion coefficients over lattice
    nx=nSubP(1); ny=nSubP(2);
    patches.cx=[repmat(a,(nx-2)/mPeriod(1),(ny-2)/mPeriod(2))
          repmat(a(1,:),1,(ny-2)/mPeriod(2))];
    patches.cy=[repmat(b,(nx-2)/mPeriod(1),(ny-2)/mPeriod(2)) ...
          repmat(b(:,1),(nx-2)/mPeriod(1),1)];

    x = reshape(patches.x,nSubP(1),1,[],1); 
    y = reshape(patches.y,1,nSubP(2),1,[]);

    disp('Check linear characteristics of the patch scheme')
    u0 = zeros(nSubP(1),nSubP(2),nPatch(1),nPatch(2));
    u0([1 end],:,:,:,:) = nan;
    u0(:,[1 end],:,:,:) = nan;
    i = find(~isnan(u0));

    disp('Construct the Jacobian, use large perturbations as linear')
    small = 1;
    jac = nan(length(i));
    sizeJacobian = size(jac)
    for j = 1:length(i)
      u = u0(:);
      u(i(j)) = u(i(j))+small;
      tmp = patchSmooth2(0,u)/small;
      jac(:,j) = tmp(i);
    end
    notSymmetric=norm(jac-jac')
    jac(abs(jac)<1e-12)=0;
    jac=sparse(jac+jac')/2;
    
    disp('Find the smallest real-part eigenvalues')
    % eigs is not reliable!!!
    evals = eigs(jac,nSmallEvals,'smallestabs','Tolerance',1e-14);
    evals = eig(jac);
    biggestImag=max(abs(imag(evals)))
    nEvals = length(evals)
    [~,k] = sort(abs(real(evals)));
    evals=evals(k);
    evals(find(abs(diff(evals))<1e-5))=[]; % remove duplicates
%    evalsWithSmallestRealPart = evals(1:2:nSmallEvals)
%    gravEvals=[gravEvals real(evalsWithSmallestRealPart)];

    disp('Effective macroscale coefficients from smallest evals')
    nev=min(ceil(min(nPatch)^2/2),nSmallEvals)
    kMax=floor((min(nPatch)-1)/2)
    evmac=real(evals(2:nev)); % remove the leading zero
    A=-evmac(1), evmac=evmac(2:end);
    for k=2:kMax % remove multiples
      [dif,j]=min(abs(evmac+A*k^2));
      if dif<1e-5*k^2,evmac(j)=[];end
    end
    B=-evmac(1), evmac=evmac(2:end);
    for k=2:kMax % remove multiples
      [dif,j]=min(abs(evmac+B*k^2));
      if dif<1e-5*k^2,evmac(j)=[];end
    end
    C=-A-B-mean(evmac(1:2))% best match next smallest evals

    disp('Quantify how macroscale diffusion matches macro-patch-spectrum')
    disp('Differences arise from micro-lattice not being continuum ...')
    disp('and from higher-orders in continuum homogenisation')
    [kx,ky]=meshgrid(-kMax:kMax);
    rates=A*kx.^2+B*ky.^2+C*kx.*ky;
    rates=sort(rates(:));
    rates(find(abs(diff(rates))<1e-5))=[];
    rates(find(abs(diff(rates))<1e-5))=[];
    evErr=real(evals(2:nev))+rates(2:nev);
    errQuartiles=quantile(abs(evErr),(1:3)/4)

end%for over different nPeriods and ratios

%gravEvals=[gravEvals std(gravEvals,0,2)]


