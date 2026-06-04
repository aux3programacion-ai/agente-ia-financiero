var PD={},ND={},PF=[],LD={};
var mcLabels={'US':'US','MEXICO':'MX','EUROPA':'EU','ASIA':'AS','GLOBAL':'GL'};
var mcColors={'US':'#00c853','MEXICO':'#ffc107','EUROPA':'#64b5f6','ASIA':'#ff5252','GLOBAL':'#ce93d8'};

function getPortafolio(){
  var ls=JSON.parse(localStorage.getItem('pf_local')||'[]');
  // ls format: ["TICKER"] or [{"ticker":"TICKER","cantidad":N}]
  var c=[];
  PF.forEach(function(item){
    if(typeof item==='string')c.push({ticker:item,cantidad:0});
    else if(item&&item.ticker)c.push({ticker:item.ticker.toUpperCase(),cantidad:item.cantidad||0});
  });
  ls.forEach(function(item){
    var ticker, cantidad=0;
    if(typeof item==='string'){ticker=item;cantidad=0;}
    else if(item&&item.ticker){ticker=item.ticker.toUpperCase();cantidad=item.cantidad||0;}
    if(ticker&&!c.some(function(e){return e.ticker===ticker}))c.push({ticker:ticker,cantidad:cantidad});
  });
  return c;
}

function getPortafolioTickers(){
  return getPortafolio().map(function(e){return e.ticker});
}

function getCantidad(ticker){
  var p=getPortafolio();
  for(var i=0;i<p.length;i++){if(p[i].ticker===ticker)return p[i].cantidad}
  return 0;
}

function agregarAlPortafolio(){
  var i=document.getElementById('pfInput');
  var q=document.getElementById('pfQty');
  var t=i.value.trim().toUpperCase();
  var cant=parseInt(q.value)||0;
  i.value='';q.value='';
  if(!t){i.placeholder='Ingresa un ticker';setTimeout(function(){i.placeholder='Ej: NVDA, AAPL, MSFT'},2000);return}
  var existing=getPortafolioTickers();
  if(existing.indexOf(t)>-1){i.placeholder='Ya esta en tu portafolio';setTimeout(function(){i.placeholder='Ej: NVDA, AAPL, MSFT'},2000);return}
  var ls=JSON.parse(localStorage.getItem('pf_local')||'[]');
  if(cant>0){ls.push({ticker:t,cantidad:cant})}else{ls.push(t)}
  localStorage.setItem('pf_local',JSON.stringify(ls));
  renderPortfolio();
  if(!PD[t]&&!LD[t])fetchLivePrice(t);
}

function eliminarDelPortafolio(t){
  if(getPortafolioTickers().indexOf(t)>-1&&PF.some(function(e){return(typeof e==='string'?e:e.ticker)===t})){
    document.getElementById('pfStatus').innerHTML='Para eliminar edita Datos/portafolio_usuario.json en GitHub';return
  }
  var ls=JSON.parse(localStorage.getItem('pf_local')||'[]');
  ls=ls.filter(function(item){
    var ticker=typeof item==='string'?item:item.ticker;
    return ticker!==t;
  });
  localStorage.setItem('pf_local',JSON.stringify(ls));
  renderPortfolio();
}

function actualizarCantidad(t,delta){
  var ls=JSON.parse(localStorage.getItem('pf_local')||'[]');
  // Check if this ticker is synced (in PF)
  var enPF=false;
  PF.forEach(function(item){
    var ticker=typeof item==='string'?item:item.ticker;
    if(ticker===t)enPF=true;
  });
  if(enPF){
    document.getElementById('pfStatus').innerHTML='Para cambiar cantidad edita Datos/portafolio_usuario.json en GitHub';return
  }
  ls=ls.map(function(item){
    var ticker=typeof item==='string'?item:item.ticker;
    if(ticker===t){
      if(typeof item==='string')return {ticker:t,cantidad:Math.max(0,delta)};
      else return {ticker:t,cantidad:Math.max(0,item.cantidad+delta)};
    }
    return item;
  });
  localStorage.setItem('pf_local',JSON.stringify(ls));
  renderPortfolio();
}

function definirCantidad(t,val){
  var ls=JSON.parse(localStorage.getItem('pf_local')||'[]');
  var enPF=false;
  PF.forEach(function(item){
    var ticker=typeof item==='string'?item:item.ticker;
    if(ticker===t)enPF=true;
  });
  if(enPF){
    document.getElementById('pfStatus').innerHTML='Para cambiar cantidad edita Datos/portafolio_usuario.json en GitHub';return
  }
  var nuevaCant=Math.max(0,parseInt(val)||0);
  ls=ls.map(function(item){
    var ticker=typeof item==='string'?item:item.ticker;
    if(ticker===t)return {ticker:t,cantidad:nuevaCant};
    return item;
  });
  localStorage.setItem('pf_local',JSON.stringify(ls));
  renderPortfolio();
}

function yahooUrl(t){return 'https://query1.finance.yahoo.com/v8/finance/chart/'+encodeURIComponent(t)+'?interval=1d&range=5d';}
function parseYahooResponse(d,t){
  try{
    var m=d.chart.result[0].meta;
    var p=m.regularMarketPrice;var qc=d.chart.result[0].indicators.quote[0].close.filter(function(v){return v!==null});
    var pp=qc.length>0?qc[qc.length-1]:p;var ch=p-pp;var pt=pp>0?((p-pp)/pp*100).toFixed(2):'0.00';
    LD[t]={p:(p||0),ch:(ch||0),pc:parseFloat(pt),pr:50,cf:50,tg:p||0,an:'Precio en vivo via Yahoo Finance',nm:t,sc:'Global',ph:0,mc:'GLOBAL'};
    renderPortfolio();
  }catch(e){fetchYahooDirect(t)}
}
function fetchYahooDirect(t){
  fetch(yahooUrl(t)).then(function(r){if(!r.ok)throw Error();return r.json()}).then(function(d){parseYahooResponse(d,t)}).catch(function(){LD[t]={p:0,ch:0,pc:0,pr:50,cf:50,tg:0,an:'Ticker no encontrado - verifica el simbolo',nm:t,sc:'Global',ph:0,mc:'GLOBAL'};renderPortfolio()});
}
var PROXIES=[
  'https://api.allorigins.win/raw?url=',
  'https://corsproxy.io/?url=',
  'https://api.codetabs.com/v1/proxy?quest=',
  'https://thingproxy.freeboard.io/fetch/'
];
function tryProxies(t,i){
  if(i>=PROXIES.length){fetchYahooDirect(t);return}
  fetch(PROXIES[i]+encodeURIComponent(yahooUrl(t)))
    .then(function(r){if(!r.ok)throw Error();return r.json()})
    .then(function(d){parseYahooResponse(d,t)})
    .catch(function(){tryProxies(t,i+1)});
}
function fetchLivePrice(t){
  LD[t]={p:0,ch:0,pc:0,pr:50,cf:50,tg:0,an:'Cargando...',nm:t,sc:'Global',ph:0,mc:'GLOBAL'};
  renderPortfolio();
  tryProxies(t,0);
}

function tickerCard(t,d,isLive){
  var cc=d.ch>=0?'#00c853':'#ff5252';
  var sg=d.ch>=0?'+':'';
  var delB='<button class="del" onclick="eliminarDelPortafolio(\''+t+'\')">&times;</button>';
  var cantidad=getCantidad(t);
  var cantHtml='';
  if(cantidad>0)cantHtml='<div class="qty" style="font-size:10px;color:#9ca3af">x'+cantidad+' acc</div>';

  // Badge: check if in synced PF (consider both formats)
  var enPF=false;
  PF.forEach(function(item){
    var ticker=typeof item==='string'?item:item.ticker;
    if(ticker===t)enPF=true;
  });
  var badge=enPF?'<span style="font-size:8px;background:#1b5e20;color:#fff;padding:1px 5px;border-radius:3px;margin-left:6px">S</span>':'<span style="font-size:8px;background:#e65100;color:#fff;padding:1px 5px;border-radius:3px;margin-left:6px">L</span>';

  var srcBadge=isLive?'<span style="font-size:8px;background:#1565c0;color:#fff;padding:1px 5px;border-radius:3px;margin-left:6px">YF</span>':'';
  var mc=d.mc||'GLOBAL';
  var mcC=mcColors[mc]||'#9ca3af';
  var mcL=mcLabels[mc]||mc;
  var mktBadge='<span style="font-size:8px;background:'+mcC+';color:#0a0b0e;padding:1px 5px;border-radius:3px;margin-left:4px;font-weight:800">'+mcL+'</span>';
  var header='<div class="tk">'+t+mktBadge+badge+srcBadge+cantHtml+'</div>';

  // Cantidad controls (only for local tickers)
  if(!enPF&&!isLive){
    var qtyEdit='<div style="margin-top:2px;font-size:10px;color:#6b7280">'+
      '<button onclick="definirCantidad(\''+t+'\','+(cantidad-1)+')" style="background:none;border:1px solid #37474f;color:#9ca3af;cursor:pointer;padding:0 4px;font-size:10px;border-radius:2px">-</button>'+
      ' <input type="number" value="'+cantidad+'" min="0" onchange="definirCantidad(\''+t+'\',this.value)" style="width:40px;background:#1e293b;border:1px solid #37474f;color:#e2e8f0;text-align:center;font-size:10px;padding:1px;border-radius:2px">'+
      ' <button onclick="definirCantidad(\''+t+'\','+(cantidad+1)+')" style="background:none;border:1px solid #37474f;color:#9ca3af;cursor:pointer;padding:0 4px;font-size:10px;border-radius:2px">+</button>'+
      ' <span style="color:#4b5563">acc</span></div>';
  } else {
    qtyEdit='';
  }

  if(isLive){
    var prc=d.p>0?'<div class="pr" style="color:'+cc+'">$'+d.p.toFixed(2)+' <span style="font-size:10px;font-weight:400">'+sg+d.pc+'%</span></div>':'<div class="pr" style="color:#6b7280">'+d.an+'</div>';
    return '<div class="pfc">'+delB+header+'<div class="nm" style="color:#9ca3af">Precio en vivo</div>'+prc+'<div class="scm"><span class="sb" style="background:#37474f">Global</span></div>'+qtyEdit+'</div>';
  }
  var pbc=d.pr>=65?'pb-h':d.pr>=58?'pb-m':'pb-l';
  var ndt=ND[t]||null;
  var nhtml='';
  if(ndt){
    var scNum=typeof ndt.sc==='number'?ndt.sc:0;
    var nsc=scNum>=0.01?'positivo':scNum<=-0.01?'negativo':'neutral';
    var nco=scNum>=0.3?'#00c853':scNum<=-0.3?'#ff5252':'#ffc107';
    nhtml='<div class="ns"><div class="nl">Noticia</div><div class="nt">'+ndt.t.substring(0,80)+'</div><div class="nsm" style="color:'+nco+'">'+nsc+' ('+scNum.toFixed(2)+')</div></div>';
  }
  var sbg=d.sc=='Semiconductores'?'#1565c0':d.sc=='Servidores IA'?'#e65100':d.sc=='Software IA'?'#00695c':d.sc=='Ciberseguridad'?'#b71c1c':d.sc=='Consumer Tech'?'#4a148c':'#37474f';
  var priceStr=typeof d.p==='number'?d.p.toFixed(2):d.p;
  var pctStr=typeof d.pc==='number'?d.pc.toFixed(2):d.pc;
  var tgStr=typeof d.tg==='number'?d.tg.toFixed(2):d.tg;
  return '<div class="pfc">'+delB+header+'<div class="nm">'+d.nm+'</div><div class="pr" style="color:'+cc+'">$'+priceStr+' <span style="font-size:10px;font-weight:400">'+sg+pctStr+'%</span></div><div class="scm"><span class="sb" style="background:'+sbg+'">'+d.sc.substring(0,12)+'</span><span style="color:#00c853;font-size:11px">$'+tgStr+'</span></div><div style="margin-top:4px"><div class="pb"><div class="pbb"><div class="pbf '+pbc+'" style="width:'+d.pr+'%"></div></div><span class="pt" style="color:'+cc+'">'+d.pr+'%</span></div></div>'+nhtml+qtyEdit+'</div>';
}

function renderPortfolio(){
  var p=getPortafolio();
  var lst=document.getElementById('pfList');
  var sts=document.getElementById('pfStats');
  if(p.length===0){lst.innerHTML='<div style="color:#4b5563;font-size:12px;grid-column:1/-1;text-align:center;padding:20px">Agrega acciones arriba. Edita Datos/portafolio_usuario.json en GitHub.</div>';sts.innerHTML='';document.getElementById('pfStatus').innerHTML='';return}
  var html='',sp=0,sc=0,sm=0,cn=0;
  p.forEach(function(e){
    var t=e.ticker;
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
  // Count total shares
  var totalShares=0;
  p.forEach(function(e){totalShares+=e.cantidad||0});
  sts.innerHTML='<div class="pv"><div class="l">Activos</div><div class="v">'+p.length+'</div></div>'+
    '<div class="pv"><div class="l">Acciones</div><div class="v">'+(totalShares>0?totalShares:'--')+'</div></div>'+
    '<div class="pv"><div class="l">Prob</div><div class="v" style="color:'+(promP>=65?'#00c853':promP>=55?'#ffc107':'#ff5252')+'">'+(sp>0?promP+'%':'--')+'</div></div>'+
    '<div class="pv"><div class="l">Conf</div><div class="v">'+(sc>0?promC+'%':'--')+'</div></div>'+
    '<div class="pv"><div class="l">Target</div><div class="v" style="color:#00c853">'+(sm>0?'$'+promT:'--')+'</div></div>';
  document.getElementById('pfStatus').innerHTML='<span style="font-size:10px;color:#4b5563">Sincronizado via GitHub | Edita Datos/portafolio_usuario.json en GitHub</span>';
}

renderPortfolio();
