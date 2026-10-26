let runtime;
async function initialize(){
 importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js','https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js');
 ort.env.wasm.numThreads=1;
 ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
 const py=await loadPyodide({indexURL:'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/'});
 await py.loadPackage(['numpy','pillow']);
 await py.runPythonAsync(await (await fetch('./prepare.py')).text());
 const model=await (await fetch('./model-v2.json')).json();
 const session=await ort.InferenceSession.create('https://raw.githubusercontent.com/metrix0/redes-neurais-pratos/main/models/resnet-int8.onnx',{executionProviders:['wasm']});
 return {py,model,session};
}
self.onmessage=async({data})=>{
 try{
 if(!runtime)runtime=initialize().catch(e=>{runtime=null;throw e;});
 const {py,model,session}=await runtime;
 py.globals.set('image_bytes',new Uint8Array(data.bytes));
 const array=await py.runPythonAsync('prepare(bytes(image_bytes.to_py()))');
 const buffer=array.getBuffer('f32');
 const tensor=new ort.Tensor('float32',new Float32Array(buffer.data),[1,3,224,224]);
 buffer.release();array.destroy();py.globals.delete('image_bytes');
 const output=await session.run({image:tensor});
 const f=output.features.data;
 let z=model.intercept;
 for(let i=0;i<f.length;i++)z+=(f[i]-model.mean[i])/model.scale[i]*model.coef[i];
 const price=Math.max(0,Math.expm1(z));
 if(!Number.isFinite(price))throw Error('Não foi possível estimar o preço.');
 self.postMessage({id:data.id,price});
 }catch(e){self.postMessage({id:data.id,error:e.message});}
};
