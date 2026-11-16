import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runDaily } from '../nightly.js';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const cloudDir=path.join(root,'cloud');
const file=path.join(cloudDir,'daily.json');
const config=JSON.parse(await fs.readFile(path.join(root,'config/research-topics.json'),'utf8'));
if(config.dailyCount!==3||config.recentCount!==2||config.classicCount!==1)throw Error('日报配置必须为两篇近期和一篇经典');
await fs.mkdir(cloudDir,{recursive:true});
let data={};try{data=JSON.parse(await fs.readFile(file,'utf8'))}catch{}
data.interests=config.interests;
await fs.writeFile(file,JSON.stringify(data,null,2));
process.env.RESEARCH_DATA_FILE=file;
const result=await runDaily({ignoreWindow:true});
if(!['complete','already_done'].includes(result.status))throw Error(`日报未完成：${result.status}`);
const saved=JSON.parse(await fs.readFile(file,'utf8'));
if(saved.dailyFeed?.papers?.length!==3)throw Error('云端日报不是三篇');
console.log(JSON.stringify({status:result.status,day:result.day,count:3}));
