var PD={},ND={},PF=[],LD={};

function getPortafolio(){
  var ls=JSON.parse(localStorage.getItem('pf_local')||'[]');
  var c=PF.slice();
  ls.forEach(function(t){if(c.indexOf(t)===-1)c.push(t)});
  return c;
}

function agregarAlPortafolio(){
  var i=document.getElementById('pfInput');
  var t=i.value.trim().toUpperCase();
  i.value='';
  if(!t){i.placeholder='Ingresa un ticker';setTimeout(function(){i.placeholder='Ej: NVDA, AAPL, MSFT, SAP.DE, WMT.MX'},2000);return}
  if(PF.indexOf(t)>-1){i.placeholder='Ya sincronizado';setTimeout(function(){i.placeholder='Ej: NVDA, AAPL, MSFT, SAP.DE, WMT.MX'},2000);return}
  var ls=JSON.parse(localStorage.getItem('pf_local')||'[]');
  if(ls.indexOf(t)===-1){ls.push(t);localStorage.setItem('pf_local',JSON.stringify(ls))}
  renderPortfolio();
  if(!PD[t]&&!LD[t])fetchLivePrice(t);
}

function eliminarDelPortafolio(t){
  if(PF.indexOf(t)>-1){document.getElementById('pfStatus').innerHTML='Para eliminar edita Datos/portafolio_usuario.json en GitHub';return}
  var ls=JSON.parse(localStorage.getItem('pf_local')||'[]');
  var idx=ls.indexOf(t);
  if(idx>-1){ls.splice(idx,1);localStorage.setItem('pf_local',JSON.stringify(ls))}
  renderPortfolio();
}

function fetchLivePrice(t){
  LD[t]={p:0,ch:0,pc:0,pr:50,cf:50,tg:0,an:'Cargando...',nm:t,sc:'Global',ph:0};
  renderPortfolio();
  fetch('https://query1.finance.yahoo.com/v8/finance/chart/'+encodeURIComponent(t)+'?interval=1d&range=5d')
    .then(function(r){if(!r.ok)throw Error('HTTP '+r.status);return r.json()})
    .then(function(d){
      var m=d.chart.result[0].meta;
      var p=m.regularMarketPrice,pc=m.chart.result[0].indicators.quote[0].close.filter(function(v){return v!==null});
      var pp=pc.length>0?pc[pc.length-1]:p;
      var ch=p-pp;
      var pt=pp>0?((p-pp)/pp*100).toFixed(2):'0.00';
      LD[t]={p:(p||0),ch:(ch||0),pc:parseFloat(pt),pr:50,cf:50,tg:p||0,an:'Precio en vivo via Yahoo Finance',nm:t,sc:'Global',ph:0};
      renderPortfolio();
    })
    .catch(function(){
      LD[t]={p:0,ch:0,pc:0,pr:50,cf:50,tg:0,an:'No disponible - verifica el ticker',nm:t,sc:'Global',ph:0};
      renderPortfolio();
    });
}

function tickerCard(t,d,isLive){
  var cc=d.ch>=0?'#00c853':'#ff5252';
  var sg=d.ch>=0?'+':'';
  var delB='<button class="del" onclick="eliminarDelPortafolio(\''+t+'\')">&times;</button>';
  var badge=PF.indexOf(t)>-1?'<span style="font-size:8px;background:#1b5e20;color:#fff;padding:1px 5px;border-radius:3px;margin-left:6px">S</span>':'<span style="font-size:8px;background:#e65100;color:#fff;padding:1px 5px;border-radius:3px;margin-left:6px">L</span>';
  var srcBadge=isLive?'<span style="font-size:8px;background:#1565c0;color:#fff;padding:1px 5px;border-radius:3px;margin-left:6px">YF</span>':'';
  var header='<div class="tk">'+t+badge+srcBadge+'</div>';
  if(isLive){
    var prc=d.p>0?'<div class="pr" style="color:'+cc+'">$'+d.p.toFixed(2)+' <span style="font-size:10px;font-weight:400">'+sg+d.pc+'%</span></div>':'<div class="pr" style="color:#6b7280">'+d.an+'</div>';
    return '<div class="pfc">'+delB+header+'<div class="nm" style="color:#9ca3af">Precio en vivo</div>'+prc+'<div class="scm"><span class="sb" style="background:#37474f">Global</span></div></div>';
  }
  var pbc=d.pr>=65?'pb-h':d.pr>=58?'pb-m':'pb-l';
  var ndt=ND[t]||null;
  var nhtml='';
  if(ndt){
    var nsc=ndt.sc>=0?'positivo':'negativo';
    var nco=ndt.sc>=0.3?'#00c853':ndt.sc<-0.3?'#ff5252':'#ffc107';
    nhtml='<div class="ns"><div class="nl">Noticia</div><div class="nt">'+ndt.t.substring(0,80)+'</div><div class="nsm" style="color:'+nco+'">'+nsc+' ('+ndt.sc+')</div></div>';
  }
  var sbg=d.sc=='Semiconductores'?'#1565c0':d.sc=='Servidores IA'?'#e65100':d.sc=='Software IA'?'#00695c':d.sc=='Ciberseguridad'?'#b71c1c':d.sc=='Consumer Tech'?'#4a148c':'#37474f';
  return '<div class="pfc">'+delB+header+'<div class="nm">'+d.nm+'</div><div class="pr" style="color:'+cc+'">$'+d.p+' <span style="font-size:10px;font-weight:400">'+sg+d.pc+'%</span></div><div class="scm"><span class="sb" style="background:'+sbg+'">'+d.sc.substring(0,12)+'</span><span style="color:#00c853;font-size:11px">$'+d.tg+'</span></div><div style="margin-top:4px"><div class="pb"><div class="pbb"><div class="pbf '+pbc+'" style="width:'+d.pr+'%"></div></div><span class="pt" style="color:'+cc+'">'+d.pr+'%</span></div></div>'+nhtml+'</div>';
}

function renderPortfolio(){
  var p=getPortafolio();
  var lst=document.getElementById('pfList');
  var sts=document.getElementById('pfStats');
  if(p.length===0){lst.innerHTML='<div style="color:#4b5563;font-size:12px;grid-column:1/-1;text-align:center;padding:20px">Agrega acciones arriba. Edita Datos/portafolio_usuario.json en GitHub.</div>';sts.innerHTML='';document.getElementById('pfStatus').innerHTML='';return}
  var html='',sp=0,sc=0,sm=0,cn=0;
  p.forEach(function(t){
    var d=PD[t]||LD[t];
    if(!d){html+='<div class="pfc"><button class="del" onclick="eliminarDelPortafolio(\''+t+'\')">&times;</button><div class="tk">'+t+'</div><div class="nm" style="color:#6b7280">Cargando precio...</div></div>';if(!LD[t])fetchLivePrice(t);return}
    cn++;
    if(PD[t]){sp+=d.pr;sc+=d.cf;sm+=d.tg}else{sp+=50}
    html+=tickerCard(t,d,!PD[t]);
  });
  lst.innerHTML=html;
  var promP=cn>0?Math.round(sp/cn):0;
  var promC=cn>0?Math.round(sc/cn):0;
  var promT=cn>0?Math.round(sm/cn):0;
  sts.innerHTML='<div class="pv"><div class="l">Activos</div><div class="v">'+p.length+'</div></div><div class="pv"><div class="l">Prob</div><div class="v" style="color:'+(promP>=65?'#00c853':promP>=55?'#ffc107':'#ff5252')+'">'+(sp>0?promP+'%':'--')+'</div></div><div class="pv"><div class="l">Conf</div><div class="v">'+(sc>0?promC+'%':'--')+'</div></div><div class="pv"><div class="l">Target</div><div class="v" style="color:#00c853">'+(sm>0?'$'+promT:'--')+'</div></div>';
  document.getElementById('pfStatus').innerHTML='<span style="font-size:10px;color:#4b5563">Sincronizado via GitHub | Edita Datos/portafolio_usuario.json en GitHub</span>';
}

renderPortfolio();
