%Simulate heterogeneous diffusion in 2D on patches as an
%example application of patches in space. Here the
%microscale is of known period so we interpolate
%next-to-edge values to get opposite edge values. Then
%explore the Jacobian and eigenvalues.  JEB, May 2020
%!TEX root = ../Doc/eqnFreeDevMan.tex
%{
\section{\texttt{homoDiffEdgy2}: computational homogenisation of a 2D diffusion by simulation on small patches}
\label{sec:homoDiffEdgy2}
%\localtableofcontents

This section extends the 1D code discussed in \cref{sec:homoDiffEdgy2} to 2D.

%}
clear all
mPeriod = [3 3]
% set random diffusion coefficients
rng('default');
rng(1); %
cHetr=exp(1*randn([mPeriod,2]));
cHetr = cHetr*mean(1./cHetr(:)) % normalise
nPeriodsPatch=[1 1]
time=0:0.01:0.5; % for arbitrary times set intial and end time only, e.g, [0 0.2]

global patches
nPatch = [16 16]
ratio = [0.1 0.1];
nSubP = [nPeriodsPatch(1)*mPeriod(1)+2 nPeriodsPatch(2)*mPeriod(2)+2]
patches.EdgyInt = 1; % one to use edges for interpolation
patches.EdgyEns=0; % one for ensemble of configurations
if patches.EdgyEns 
    patches.EdgyInt=1; % EdgyEns=1 implies EdgyInt=1     
   % nSubP = [3 3] % > [2 2]; when EdgyEns=1, nSubP need not depend on mPeriod
end
configPatches2(@heteroDiff2,[0 2*pi 0 2*pi],nan,nPatch ...
    ,0,ratio,nSubP);
%{
\end{matlab}

Replicate the heterogeneous coefficients across the width of
each patch. For \verb|patches.EdgyEns| an ensemble of configurations 
is constructed.
\begin{matlab}
%}
if patches.EdgyEns   
   shiftx=mod(bsxfun(@plus,0:(mPeriod(1)-1),(0:(nSubP(1)-2))'),mPeriod(1))+1;
   shifty=mod(bsxfun(@plus,0:(mPeriod(2)-1),(0:(nSubP(2)-2))'),mPeriod(2))+1;
   patches.cx=nan(nSubP(1)-1,nSubP(2)-2,mPeriod(1)*mPeriod(2));
   patches.cy=nan(nSubP(1)-2,nSubP(2)-1,mPeriod(1)*mPeriod(2));
   for p=1:mPeriod(1)
      for  q=1:mPeriod(2)
          patches.cx(:,:,(p-1)*mPeriod(2)+q)=cHetr(shiftx(:,p),shifty(1:(end-1),q),1);
          patches.cy(:,:,(p-1)*mPeriod(2)+q)=cHetr(shiftx(1:(end-1),p),shifty(:,q),2);
      end
   end
   patches.cx=permute(repmat(patches.cx,[1,1,1,nPatch]),[1,2,4,5,3]);
   patches.cy=permute(repmat(patches.cy,[1,1,1,nPatch]),[1,2,4,5,3]);
   % need to specify how configurations are coupled 
   patches.le=mod((0:(mPeriod(1)*mPeriod(2)-1))+mPeriod(2)*rem(nSubP(1)-2,mPeriod(1)),mPeriod(1)*mPeriod(2))+1;
   patches.ri=mod((0:(mPeriod(1)*mPeriod(2)-1))-mPeriod(2)*rem(nSubP(1)-2,mPeriod(1)),mPeriod(1)*mPeriod(2))+1;
   patches.bo=mod((1:(mPeriod(1)*mPeriod(2)))+nSubP(2)-3,mPeriod(2))+mPeriod(2)*floor((0:(mPeriod(1)*mPeriod(2)-1))/mPeriod(2))+1;
   patches.to=mod((1:(mPeriod(1)*mPeriod(2)))-nSubP(2)+1,mPeriod(2))+mPeriod(2)*floor((0:(mPeriod(1)*mPeriod(2)-1))/mPeriod(2))+1;
else  
   patches.cx=[repmat(cHetr(:,:,1),[(nSubP-2)./mPeriod,1]);repmat(cHetr(1,:,1),[1,(nSubP(2)-2)/mPeriod(2)])];
   patches.cy=[repmat(cHetr(:,:,2),[(nSubP-2)./mPeriod,1]),repmat(cHetr(:,1,2),[(nSubP(1)-2)/mPeriod(1),1])];
end
%{
\end{matlab}

\paragraph{Simulate}
Set the initial conditions of a simulation.
\begin{matlab}
%}

n_modes = 10;
max_wave_number = 5;
k1 = randi(max_wave_number,n_modes);
k2 = randi(max_wave_number,n_modes);
p1 = pi * ( randi(2,n_modes)-1);
p2 = pi * ( randi(2,n_modes)-1);
a = 2*rand(n_modes)-1;
Vp = zeros(length(patches.x(:)));
for j=1:n_modes
    Vp = Vp + a(j) .* sin(k1(j)*patches.x(:) + p1(j)) .* sin(k2(j)*patches.y(:)' + p2(j));
end

if patches.EdgyEns 
   u0=repmat(permute(reshape(Vp,[nSubP(1),nPatch(1),nSubP(2),nPatch(2)]),[1,3,2,4]),1,1,1,1,mPeriod(1)*mPeriod(2));
   % u0=repmat(permute(reshape((cos(patches.x(:))*sin(patches.y(:))'),[nSubP(1),nPatch(1),nSubP(2),nPatch(2)]),[1,3,2,4])+0.2*randn([nSubP,nPatch]) ...
   % ,1,1,1,1,mPeriod(1)*mPeriod(2));
else
   u0=permute(reshape(Vp,[nSubP(1),nPatch(1),nSubP(2),nPatch(2)]),[1,3,2,4]); 
   %u0=permute(reshape((cos(patches.x(:))*sin(patches.y(:))'),[nSubP(1),nPatch(1),nSubP(2),nPatch(2)]),[1,3,2,4])+0.2*randn([nSubP,nPatch]);  
end
%{
\end{matlab}
Integrate using standard stiff integrators.
\begin{matlab}
%}

if ~exist('OCTAVE_VERSION','builtin')
    [ts,us] = ode15s(@patchSmooth2, time, u0(:));
else % octave version
    [ts,us] = odeOcts(@patchSmooth2, time, u0(:));
end

% compute temporal derivatives at times ts
utsim=zeros(size(us));
for i=1:length(ts)
    utsim(i,:)=patchSmooth2(ts(i),us(i,:));
end
% reshape into patch structure
usim=reshape(us,[length(ts),size(u0)]);
utsim=reshape(utsim,[length(ts),size(u0)]);
% get rid of edges
usim(:,1,:,:,:,:)=[];
usim(:,end,:,:,:,:)=[];
usim(:,:,1,:,:,:)=[];
usim(:,:,end,:,:,:)=[];
utsim(:,1,:,:,:,:)=[];
utsim(:,end,:,:,:,:)=[];
utsim(:,:,1,:,:,:)=[];
utsim(:,:,end,:,:,:)=[];


%{
\end{matlab}
Plot field solutions.
\begin{matlab}
%}

% %  % for video
% frames = struct('cdata', cell(1, length(ts)), 'colormap', cell(1, length(ts)));
% 
% disp('plot sequence of surfaces')
% figure(1), clf, colormap(hsv)
% x = patches.x; y = patches.y;
% x(end+1,:)=nan; y(end+1,:)=nan; % pad with nans
% uPad=nan([nSubP+1,nPatch]);
% upplt=nan(length(time),length(x(:)),length(y(:)));
% for i = 1:length(ts)
%     if  patches.EdgyEns
%        uPad(1:(end-1),1:(end-1),:,:) = mean(reshape(patchEdgeInt2(us(i,:)'),[nSubP,nPatch,mPeriod(1)*mPeriod(2)]),5);
%     else
%         uPad(1:(end-1),1:(end-1),:,:) = reshape(patchEdgeInt2(us(i,:)'),[nSubP,nPatch]);
%     end    
%   upplt(i,:,:) = reshape(permute(uPad,[1 3 2 4]), [numel(x) numel(y)]);
%   if i==1,
%   hsurf = surf(x(:),y(:),squeeze(upplt(i,:,:))');
%   axis([0 2*pi 0 2*pi -1.3 1.3]), view(60,40) %view(60,40)
%   xlabel('x'), ylabel('y'), zlabel('u')
%   else   set(hsurf,'ZData',squeeze(upplt(i,:,:))');
%   end
%   legend(['time = ' num2str(ts(i),2)],'Location','north')
%   caxis([0 1])
%   pause(0.1)
%   frames(i) = getframe(gcf); 
% end
% % for video
%       vw = VideoWriter('diffusion2d','MPEG-4');  %taking a guess that you intend to modify the filename each time you write a video
%       open(vw);
%       writeVideo(vw, frames);
%       close(vw);


% Find macroscale eigenvalues of patch scheme and then approximate first three coefficients of macroscale PDE

nSmallEvals=nPatch(1)*(nPatch(2)+1);
egs=nan(11,nSmallEvals);

    disp('Check linear characteristics of the patch scheme')
    u0 = zeros([nSubP, nPatch]);
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
    assert(notSymmetric<1e-7,'failed symmetry')
            
    gravEvals=[];
   
    disp('Find the smallest real-part eigenvalues')
    evals = eig(jac);
    biggestImag=max(abs(imag(evals)))
    nEvals = length(evals);
    [~,k] = sort(abs(real(evals)));
    evals=evals(k);
    evalsWithSmallestRealPart = evals(1:nSmallEvals);
    gravEvals=[gravEvals real(evalsWithSmallestRealPart)];

    disp('Effective macroscale coefficients from smallest evals')
    nev=min(ceil(min(nPatch)^2/2),nSmallEvals)
    kMax=floor((min(nPatch)-1)/2)
    evmac=real(evals(2:2:nev)); % remove the leading zero
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

global macros
macros.A=A;
macros.B=B;
macros.C=C;

% match some macroscale points with centres of patches
p=3; % 2*p points per macroscale step
macros.dx=(patches.x(1,1)+patches.x(end,1))/(2*p);
macros.dy=macros.dx;

% inital conditions, same as for patch simulation
xval=macros.dx:macros.dx:(2*pi); % p points per macroscale step;
yval=macros.dy:macros.dy:(2*pi);

Vfull = zeros(size(xval));
for j=1:n_modes
    Vfull = Vfull + a(j) .* sin(k1(j)*xval(:) + p1(j)) .* sin(k2(j)*yval(:)' + p2(j));
end
uu0=Vfull;

[tts,uus] = ode15s(@macrodiff2, time, uu0);

clf
uuplt=reshape(uus,length(tts),length(xval),length(yval));
% figure;
% frames = struct('cdata', cell(1, length(tts)), 'colormap', cell(1, length(tts)));
% for i = 1:length(tts)
%   if i==1
%   hsurf = surf(xval(:),yval(:),squeeze(uuplt(i,:,:))');
%   axis([0 2*pi 0 2*pi -1.3 1.3]), view(60,40) %view(60,40)
%   xlabel('x'), ylabel('y'), zlabel('u')
%   else   set(hsurf,'ZData', squeeze(uuplt(i,:,:))');  
%   end
%   legend(['time = ' num2str(tts(i),2)],'Location','north')
%   caxis([0 1])
%   pause(0.1)
%   frames(i) = getframe(gcf); 
% end
% % for video
%       vx = VideoWriter('diffusion2dmacro','MPEG-4');  %taking a guess that you intend to modify the filename each time you write a video
%       open(vx);
%       writeVideo(vx, frames);
%       close(vx);     
%       

%% for comparison with patch simulation
macrostp=p:(2*p):length(xval);
mac=uuplt(:,macrostp,macrostp); % macroscale at patch centres
mic=squeeze(mean(mean(usim,3),2)); % patch averages
figure(12)

subplot(2,2,1)
t=6;
surf(squeeze(abs(mac(t,:,:)-mic(t,:,:))))
title(['time = ',num2str(time(t))]);

subplot(2,2,2)
t=11;
surf(squeeze(abs(mac(t,:,:)-mic(t,:,:))))
title(['time = ',num2str(time(t))]);

subplot(2,2,3)
t=16;
surf(squeeze(abs(mac(t,:,:)-mic(t,:,:))))
title(['time = ',num2str(time(t))]);

subplot(2,2,4)
t=21;
surf(squeeze(abs(mac(t,:,:)-mic(t,:,:))))
title(['time = ',num2str(time(t))]);

print('-dpng','-painters',"compare2");

%% plot of error
dv = mic-mac;
er=[];
nom = [];
for i=1:size(dv,1)
    dv_t = dv(i,:,:).^2;
    vhom=mac(i,:,:);
    er=[er,mean(dv_t(:))];
    nom = [nom,var(vhom(:))];
end

figure(121),hold on
semilogy(time,er./nom)
