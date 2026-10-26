const $=id=>document.getElementById(id);
let busy=false,imageURL;
const status=(text,error=false)=>{
 $('status-text').textContent=text;
 $('status').classList.toggle('error',error);
};
function setBusy(value){
 busy=value;
 $('photo-card').classList.toggle('processing',value);
 $('sample-button').disabled=value;
 $('file-input').disabled=value;
 $('drop-zone').setAttribute('aria-disabled',String(value));
}
async function analyze(file,name){
 if(busy)return;
 if(file.size>10*1024*1024)return status('A foto precisa ter até 10 MB.',true);
 if(!['image/jpeg','image/png','image/webp'].includes(file.type))return status('Escolha JPG, PNG ou WEBP.',true);
 setBusy(true);
 status('Analisando a imagem com a ResNet-18…');
 try{
  const bitmap=await createImageBitmap(file);
  if(bitmap.width*bitmap.height>40000000){bitmap.close();throw Error('Use uma imagem com até 40 megapixels.');}
  bitmap.close();
  if(imageURL)URL.revokeObjectURL(imageURL);
  imageURL=URL.createObjectURL(file);
  $('dish-photo').src=imageURL;$('dish-photo').alt=name;
  $('photo-kicker').textContent='UM NOVO OLHAR SOBRE A SUA FOTO';
  $('photo-name').textContent=name==='Massa ao pesto'?name:'Seu prato, em foco.';
  const form=new FormData();form.append('file',file);
  const response=await fetch('/api/predict',{method:'POST',body:form,signal:AbortSignal.timeout(120000)});
  const result=await response.json();
  if(!response.ok)throw Error(result.detail||'Não foi possível analisar esta foto.');
  $('price').textContent=result.price.toLocaleString('pt-BR',{minimumFractionDigits:2,maximumFractionDigits:2});
  $('upload-view').hidden=true;$('result-view').hidden=false;
  status('Análise concluída. A foto não foi armazenada.');
 }catch(error){status(error.message||'Não foi possível analisar a imagem.',true);}
 finally{setBusy(false);}
}
$('file-input').addEventListener('change',e=>{if(e.target.files[0])analyze(e.target.files[0],'Sua foto');e.target.value='';});
const drop=$('drop-zone');drop.tabIndex=0;
drop.addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();if(!busy)$('file-input').click();}});
['dragenter','dragover'].forEach(event=>drop.addEventListener(event,e=>{e.preventDefault();if(!busy)drop.classList.add('dragging');}));
['dragleave','drop'].forEach(event=>drop.addEventListener(event,e=>{e.preventDefault();drop.classList.remove('dragging');}));
drop.addEventListener('drop',e=>{if(e.dataTransfer.files[0])analyze(e.dataTransfer.files[0],'Sua foto');});
$('sample-button').addEventListener('click',async()=>{
 if(busy)return;
 try{const response=await fetch('./pasta.jpg');if(!response.ok)throw Error();await analyze(new File([await response.blob()],'massa.jpg',{type:'image/jpeg'}),'Massa ao pesto');}
 catch{status('Não foi possível carregar o exemplo.',true);}
});
$('reset-button').addEventListener('click',()=>{
 $('upload-view').hidden=false;$('result-view').hidden=true;
 status('Sua foto é processada sem armazenamento.');
});
