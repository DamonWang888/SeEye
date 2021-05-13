% load vonkoch
% vonkoch=vonkoch(1:510); 

% Fs=100;
% t=0:1/Fs:1;
% 
% f=5;
% 
% vonkoch=sin(2*pi*f*t);
% len = length(vonkoch);
% cw1 = cwt(vonkoch,1:10,'sym2','plot'); 
% title('Continuous Transform, absolute coefficients.');
% ylabel('Scale');
% [cw1,sc] = cwt(vonkoch,1:32,'sym2','scal');
% title('Scalogram'); 
% ylabel('Scale');

% 
% load vonkoch 					
% vonkoch=vonkoch(1:510);
% len=length(vonkoch);
% [c,l]=wavedec(vonkoch,5,'sym2');
% % Compute and reshape DWT to compare with CWT.
% cfd=zeros(5,len);
% for k=1:5
%     d=detcoef(c,l,k);
%     d=d(ones(1,2^k),:);
%     cfd(k,:)=wkeep(d(:)',len);
% end
% cfd=cfd(:);
% I=find(abs(cfd) <sqrt(eps));
% cfd(I)=zeros(size(I));
% cfd=reshape(cfd,5,len);
% % Plot DWT.
% subplot(311); plot(vonkoch); title('Analyzed signal.');
% set(gca,'xlim',[0 510]);
% subplot(312); 
% image(flipud(wcodemat(cfd,255,'row')));
% colormap(pink(255));
% set(gca,'yticklabel',[]);
% title('Discrete Transform,absolute coefficients');
% ylabel('Level');
% % Compute CWT and compare with DWT
% subplot(313);
% ccfs=cwt(vonkoch,1:32,'sym2','plot');
% title('Continuous Transform, absolute coefficients');
% set(gca,'yticklabel',[]);
% ylabel('Scale');


function[coefs]=cwt_demo1(z,fs)

Fs=186.667e6;
dt=1/Fs;
t=(0:length(z)/Fs/dt-1)*dt;

% 定义信号信息
% fs=2^6;    %采样频率
% dt=1/fs;    %时间精度
% timestart=-8;
% timeend=8;
% t=(0:(timeend-timestart)/dt-1)*dt+timestart;
% L=length(t);
% 
% z=4*sin(2*pi*linspace(6,12,L).*t);
% z=4*sin(2*pi*10*t);
%旧版本
wavename='cmor1-3'; %可变参数，分别为cmor的
% wavename='sym2'; %可变参数，分别为cmor的 
%举一个频率转尺度的例子
fmin=2;
fmax=200;
df=0.1;

f=fmin:df:fmax-df;%预期的频率
wcf=centfrq(wavename); %小波的中心频率
scal=fs*wcf./f;%利用频率转换尺度
coefs = cwt(z,scal,wavename);
figure(2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
pcolor(t,f,abs(coefs));shading interp

end


