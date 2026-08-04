// Copyright 2025 Autodesk, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pch.h"

#include "Transpiler.h"
#include <slang.h>
#include <slang-com-ptr.h>

BEGIN_AURORA

// Slang string blob.  Simple wrapper around std::string that can be used by Slang compiler.
struct StringSlangBlob : public ISlangBlob, ISlangCastable
{
    StringSlangBlob(const string& str) : _str(str) {}
    virtual ~StringSlangBlob() = default;

    // Get a pointer to the string.
    virtual SLANG_NO_THROW void const* SLANG_MCALL getBufferPointer() override
    {
        return _str.data();
    }

    // Get the length of the string.
    virtual SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() override { return _str.length(); }

    // Default queryInterface implementation for Slang type system.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(
        SlangUUID const& uuid, void** outObject) SLANG_OVERRIDE
    {
        *outObject = getInterface(uuid);

        return *outObject ? SLANG_OK : SLANG_E_NOT_IMPLEMENTED;
    }

    // Do not implement ref counting, just return 1.
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() SLANG_OVERRIDE { return 1; }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() SLANG_OVERRIDE { return 1; }

    // Default castAs implementation for Slang type system.
    void* castAs(const SlangUUID& guid) override { return getInterface(guid); }

    // Allow casting as Unknown and Blob, but nothing else.
    void* getInterface(const SlangUUID& uuid)
    {
        if (uuid == ISlangUnknown::getTypeGuid() || uuid == ISlangBlob::getTypeGuid())
        {
            return static_cast<ISlangBlob*>(this);
        }

        return nullptr;
    }

    // The actual string data.
    const string& _str;
};

// Slang filesystem that reads from a simple lookup table of strings.
struct AuroraSlangFileSystem : public ISlangFileSystem
{
    AuroraSlangFileSystem(const std::map<std::string, const std::string&>& fileText) :
        _fileText(fileText)
    {
    }
    virtual ~AuroraSlangFileSystem() = default;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL loadFile(
        char const* path, ISlangBlob** outBlob) override
    {
        // Is if we already have blob for this path, return that.
        auto iter = _fileBlobs.find(path);
        if (iter != _fileBlobs.end())
        {
            *outBlob = iter->second.get();

            return SLANG_OK;
        }

        // Read from the text file map.
        auto shaderIter = _fileText.find(path);
        if (shaderIter != _fileText.end())
        {
            // Create a blob from the text file string, add to the blob map, and return it.
            _fileBlobs[path] = make_unique<StringSlangBlob>(shaderIter->second);
            *outBlob         = _fileBlobs[path].get();
            return SLANG_OK;
        }

        return SLANG_FAIL;
    }

    // Slang type interface not needed, just return null.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(
        SlangUUID const& /* uuid*/, void** /* outObject*/) SLANG_OVERRIDE
    {
        return SLANG_E_NOT_IMPLEMENTED;
    }

    // Do not implement ref counting, just return 1.
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() SLANG_OVERRIDE { return 1; }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() SLANG_OVERRIDE { return 1; }

    // Slang type interface not needed, just return null.
    void* castAs(const SlangUUID&) override { return nullptr; }

    // Set a string directly as blob in file blobs map.
    void setSource(const string& name, const string& code)
    {
        _fileBlobs[name] = make_unique<StringSlangBlob>(code);
    }

    slang::IBlob* getSource(const string& name)
    {
        auto iter = _fileBlobs.find(name);
        return iter != _fileBlobs.end() ? iter->second.get() : nullptr;
    }

    // Map of string blobs.
    map<string, unique_ptr<StringSlangBlob>> _fileBlobs;

    // Map of file strings.
    const std::map<std::string, const std::string&>& _fileText;
};

Transpiler::Transpiler(const std::map<std::string, const std::string&>& fileText)
{
    _pFileSystem = make_unique<AuroraSlangFileSystem>(fileText);

    SlangGlobalSessionDesc desc;
    desc.enableGLSL = true;
    slang::createGlobalSession(&desc, &_pSession);
}

Transpiler::~Transpiler()
{
    _pSession->release();
}

void Transpiler::setSource(const string& name, const string& code)
{
    _pFileSystem->setSource(name, code);
}

bool Transpiler::transpileCode(const string& shaderCode, string& codeOut, string& errorOut,
    Language target, const map<string, string>& preprocessorDefines)
{
#if defined(__APPLE__)
    // TODO: We do actually want to be transpiling here eventually
    return true;
#endif
    // Dummy file name to use as container for shader code.
    const string codeFileName = "__shaderCode";

    // Set the shader code "file".
    setSource(codeFileName, shaderCode);

    // Transpile the shader code.
    bool res = transpile(codeFileName, codeOut, errorOut, target, preprocessorDefines);

    // Clear the shader source to release memory.
    setSource(codeFileName, "");

    return res;
}

bool Transpiler::transpile(const string& shaderName, string& codeOut, string& errorOut,
    Language target, const map<string, string>& preprocessorDefines)
{
    // Clear result.
    errorOut.clear();
    codeOut.clear();

    // TODO: Multithreading.
    using namespace slang;

    // Create the target description for the session.
    TargetDesc targetDesc;
    switch (target)
    {
    case Language::HLSL:
        targetDesc.format  = SLANG_HLSL;
        targetDesc.profile = _pSession->findProfile("lib_6_3");
        break;
    case Language::GLSL:
        targetDesc.format  = SLANG_GLSL;
        targetDesc.profile = _pSession->findProfile("glsl_460");
        break;
    case Language::Metal:
        targetDesc.format  = SLANG_METAL;
        targetDesc.profile = _pSession->findProfile("metallib_2_3");
        break;
    default:
        AU_FAIL("Unsupported target language for transpiler.");
        return false;
    }
    // Use standard line directives (with filename).
    targetDesc.lineDirectiveMode = SLANG_LINE_DIRECTIVE_MODE_STANDARD;
    // TODO: The buffer layout might be an issue, need to work out correct flags.
    // targetDesc.forceGLSLScalarBufferLayout = true;

    // Create compiler options.
    std::vector<CompilerOptionEntry> compilerOptions;
    compilerOptions.push_back(
        { CompilerOptionName::NoMangle, { CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr } });
    compilerOptions.push_back({ CompilerOptionName::GenerateWholeProgram,
        { CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr } });
    targetDesc.compilerOptionEntries    = compilerOptions.data();
    targetDesc.compilerOptionEntryCount = static_cast<uint32_t>(compilerOptions.size());

    // Create the session description.
    SessionDesc sessionDesc;
    sessionDesc.targets                 = &targetDesc;
    sessionDesc.targetCount             = 1;
    sessionDesc.fileSystem              = _pFileSystem.get();
    sessionDesc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
    sessionDesc.allowGLSLSyntax         = true;

    // Setup pre-defined macros.
    std::vector<PreprocessorMacroDesc> preprocessorMacros;
    for (const auto& [key, value] : preprocessorDefines)
    {
        preprocessorMacros.push_back({ key.c_str(), value.c_str() });
    }
    preprocessorMacros.push_back({ "DIRECTX", target == Language::HLSL ? "1" : "0" });
    sessionDesc.preprocessorMacros     = preprocessorMacros.data();
    sessionDesc.preprocessorMacroCount = preprocessorMacros.size();

    // Create a session representing a scope for compilation with a consistent set of compiler
    // options.
    Slang::ComPtr<ISession> session;
    _pSession->createSession(sessionDesc, session.writeRef());

    // Transpile the file.
    Slang::ComPtr<IBlob> diagnostics;
    const string fileName = shaderName + ".slang";
    Slang::ComPtr<IModule> sessionModule(session->loadModuleFromSource(shaderName.c_str(),
        fileName.c_str(), _pFileSystem->getSource(shaderName), diagnostics.writeRef()));
    if (diagnostics)
    {
        errorOut = (const char*)diagnostics->getBufferPointer();
        return false;
    }

    // Link the module to get a program
    Slang::ComPtr<IComponentType> linkedProgram;
    sessionModule->link(linkedProgram.writeRef(), diagnostics.writeRef());
    if (diagnostics)
    {
        errorOut = (const char*)diagnostics->getBufferPointer();
        return false;
    }

    // Get blob for result.
    Slang::ComPtr<ISlangBlob> outBlob;
    linkedProgram->getTargetCode(0 /* targetIndex */, outBlob.writeRef(), diagnostics.writeRef());
    if (diagnostics)
    {
        errorOut = (const char*)diagnostics->getBufferPointer();
        return false;
    }
    codeOut = (const char*)outBlob->getBufferPointer();
    return true;
}

END_AURORA
