const {Octokit} = require ("octokit")
const axios = require("axios").default
const diffParser = require("gitdiff-parser")
const {execSync} = require("child_process")
const fs = require("fs")
const path = require("path")
const dotparser = require('dotparser');

const dataset = require("../Data/dataset.json")

require('dotenv').config()

const octokit = new Octokit({
    auth: process.env.TOKEN
});


async function get_data(){

  if(!fs.existsSync(path.join(__dirname, 'repos'))){
    fs.mkdirSync(path.join(__dirname, 'repos'))
  }

  if(!fs.existsSync(path.join(__dirname, 'temp'))){
    fs.mkdirSync(path.join(__dirname, 'temp'))
  }

  let i = 0;

  for(let x of Object.keys(dataset)){

    console.log(i);
    i++;

    owner = x.split("/")[0]
    repo = x.split("/")[1].split("_")[0]
    pull_number = x.split("/")[1].split("_")[1]


    const pull_res = await octokit.request("GET /repos/{owner}/{repo}/pulls/{pull_number}", {
      owner,
      repo,
      pull_number
    })

    const {issue_url, diff_url} = pull_res.data
    
    try{
      const issue_res = await axios.get(issue_url)
      dataset[x]["issue_title"] = issue_res.data.title
    }catch(e){
      console.log(e);
      console.log("No issue associated");
      dataset[x]["issue_title"] = ""
    }

    console.log("issue title check.");

    // ---------------- ASTs ---------------------------------------
    // --------- Clone the repo if it doesn't exist ----------------

    if(!fs.existsSync(path.join(__dirname, 'repos', owner))){
      fs.mkdirSync(path.join(__dirname, 'repos', owner))
    }

    if(!fs.existsSync(path.join(__dirname, 'repos', owner, repo))){
      const owner_path = path.join(__dirname, 'repos', owner).replace(/\ /g, "\\ ")
      execSync(`cd ${owner_path} && git clone https://github.com/${owner}/${repo}.git`)
    }

    console.log("repos check.");

    // -----------------------------------------------------------

    let old_hashs = {}

    const diff_res = await axios.get(diff_url)
    const diff_output = diff_res.data
    const files = diffParser.parse(diff_output)

    for(let file of files){
      if(file.newPath.slice(-5) === ".java" && file.oldPath.slice(-5) === ".java"){

        const old_hash = file.oldRevision
        const new_hash = file.newRevision
        old_hashs[`${new_hash}`] = `${old_hash}`
      }
    }

    console.log("old and new hashes check.");

    let com_i = 0
    for(let commit_sha of Object.keys(dataset[x]["commits"])){

      console.log("commit : " + com_i);
      com_i++;
      
      const commit_res = await octokit.request('GET /repos/{owner}/{repo}/commits/{ref}', {
        owner: owner,
        repo: repo,
        ref: commit_sha.slice(1, -1)
      })

      const new_file_shas = commit_res.data["files"].filter(file => (file.filename.slice(-5) === ".java")).map(file => file.sha)
      const patches = commit_res.data["files"].filter(file => (file.filename.slice(-5) === ".java")).map(file => file.patch)

      if(dataset[x]["commits"][commit_sha]["old_asts"] === undefined){
        dataset[x]["commits"][commit_sha]["old_asts"] = []
      }
      if(dataset[x]["commits"][commit_sha]["new_asts"] === undefined){
        dataset[x]["commits"][commit_sha]["new_asts"] = []
      }

      console.log("files changed " + new_file_shas.length);

      for(let i=0; i<new_file_shas.length; i++){

        const new_file_sha = new_file_shas[i]
        const old_file_sha = old_hashs[new_file_sha.slice(0, 13)]
        const patch = patches[i]

        // new_file_sha & old_file_sha are available

        // Get the files
        const repo_path = path.join(__dirname, 'repos', owner, repo).replace(/\ /g, "\\ ")

        if(old_file_sha !== undefined){
          execSync(`cd ${repo_path} && git show ${old_file_sha} > ../../../temp/old.java`)
          execSync(`cd temp && joern-parse old.java && joern-export --repr ast --out old_ast`)
        }

        if(new_file_sha !== undefined){
          execSync(`cd ${repo_path} && git show ${new_file_sha} > ../../../temp/new.java`)
          execSync(`cd temp && joern-parse new.java && joern-export --repr ast --out new_ast`)
        }

        console.log("retrieved java file " + i);

        // patch contails all the chunks where the changes happened in a file.
        const hunk_headers = patch.split('\n').filter(line => line.startsWith("@@"))

        console.log("changes in file " + i + " : " + hunk_headers.length);

        let j = 0;
        for(let hunk_header of hunk_headers){

          const function_name = hunk_header.split("@").slice(-1)[0].split("(")[0].split(" ").slice(-1)[0]

          if(old_file_sha === undefined)
            dataset[x]["commits"][commit_sha]["old_asts"].push({})
          else{
            for(let dotfile of fs.readdirSync(path.join('temp', 'old_ast'))){
              const dotfile_contents = fs.readFileSync(path.join('temp', 'old_ast', dotfile))
              const ast = dotparser(dotfile_contents.toString())
              if(ast[0].id === function_name){
                const custom_ast = get_custom_ast(ast[0])
                dataset[x]["commits"][commit_sha]["old_asts"].push(custom_ast)
                break
              }
            }
          }

          if(new_file_sha === undefined)
            dataset[x]["commits"][commit_sha]["new_asts"].push({})
          else{
            for(let dotfile of fs.readdirSync(path.join('temp', 'new_ast'))){
              const dotfile_contents = fs.readFileSync(path.join('temp', 'new_ast', dotfile))
              const ast = dotparser(dotfile_contents.toString())
              if(ast[0].id === function_name){
                const custom_ast = get_custom_ast(ast[0])
                dataset[x]["commits"][commit_sha]["new_asts"].push(custom_ast)
                break
              }
            }
          }
          console.log("change " + j + " done.");
          j++;
        }
        execSync(`cd temp && rm -r *`)
      }
    }
  }
  fs.writeFileSync('../Data/dataset_aug.json', JSON.stringify(dataset))
}

get_data()



function get_custom_ast(ast){

  custom_ast = {}

  /**
   * Structure of the AST in JSON format:
   * custom_ast = {id: {label: "sdbv", children: [id1, id2....]}}
   */

  for(let child of ast.children){

    if(child.type === 'node_stmt'){
        node_id = parseInt(child.node_id.id)
        
        let label = child.attr_list[0].eq.value
        label = label.split('(')[1].split(',')[0]
        
        custom_ast[node_id] = {label: label, children: []}

    }else if(child.type === 'edge_stmt'){
      const u = parseInt(child.edge_list[0].id)
      const v = parseInt(child.edge_list[1].id)
      custom_ast[u].children.push(v)

    }
  }

  return custom_ast

}
